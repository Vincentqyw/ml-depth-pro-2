import MetalKit
import SwiftUI
import UIKit

struct MetalView: UIViewRepresentable {
    let engine: DepthEngine
    let mode: Int
    let blend: Float
    let compare: Bool

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: engine.gpu.device)
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        view.isOpaque = true
        view.backgroundColor = .black
        let renderer: Renderer
        do {
            renderer = try Renderer(gpu: engine.gpu, engine: engine)
        } catch {
            fatalError("Cannot create the depth renderer: \(error)")
        }
        renderer.mode = mode
        renderer.blend = blend
        renderer.compare = compare
        context.coordinator.renderer = renderer
        view.delegate = renderer
        engine.setOnFrame { [weak view] in
            DispatchQueue.main.async { view?.setNeedsDisplay() }
        }
        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {
        context.coordinator.renderer.mode = mode
        context.coordinator.renderer.blend = blend
        context.coordinator.renderer.compare = compare
        view.setNeedsDisplay()
    }

    final class Coordinator {
        var renderer: Renderer!
    }
}

private struct RenderUniforms {
    var scaleNearFar: SIMD3<Float>
    var mode: Float
    var blend: Float
    var uvScale: SIMD2<Float>
    var quarterTurns: Int32
}

private struct MetersUniforms {
    var nearFar: SIMD2<Float>
    var uvScale: SIMD2<Float>
    var cropScale: SIMD2<Float>
    var cropOffset: SIMD2<Float>
    var quarterTurns: Int32
}

extension UIInterfaceOrientation {
    /// Quarter turns from the camera's native landscape-right buffer to this interface.
    var quarterTurns: Int {
        switch self {
        case .portrait: 1
        case .landscapeLeft: 2
        case .portraitUpsideDown: 3
        default: 0
        }
    }
}

@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    private let gpu: GPU
    private let engine: DepthEngine
    private let depthPipeline: MTLRenderPipelineState
    private let metersPipeline: MTLRenderPipelineState
    var mode = 2
    var blend: Float = 0.8
    var compare = false

    init(gpu: GPU, engine: DepthEngine) throws {
        self.gpu = gpu
        self.engine = engine
        depthPipeline = try Self.pipeline(gpu: gpu, fragment: "display")
        metersPipeline = try Self.pipeline(gpu: gpu, fragment: "display_meters")
        super.init()
    }

    private static func pipeline(gpu: GPU, fragment: String) throws -> MTLRenderPipelineState {
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = gpu.library.makeFunction(name: "fullscreen")
        descriptor.fragmentFunction = gpu.library.makeFunction(name: fragment)
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        return try gpu.device.makeRenderPipelineState(descriptor: descriptor)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let commands = gpu.queue.makeCommandBuffer()
        else { return }
        guard let pool = engine.currentPool, let slot = pool.acquireLatest() else {
            if let encoder = commands.makeRenderCommandEncoder(descriptor: pass) { encoder.endEncoding() }
            commands.present(drawable)
            commands.commit()
            return
        }
        guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else {
            pool.release(slot)
            return
        }
        let turns = view.window?.windowScene?.interfaceOrientation.quarterTurns ?? 0
        let size = (width: Double(drawable.texture.width), height: Double(drawable.texture.height))
        if compare {
            // Portrait stacks the two panes; landscape puts them side by side.
            let stacked = !turns.isMultiple(of: 2)
            let pane = stacked
                ? (x: 0.0, y: 0.0, width: size.width, height: size.height / 2)
                : (x: 0.0, y: 0.0, width: size.width / 2, height: size.height)
            drawDepth(slot, encoder: encoder, turns: turns, pane: pane, mode: 2)
            drawARKit(
                slot, encoder: encoder, turns: turns,
                pane: stacked
                    ? (x: 0.0, y: size.height / 2, width: pane.width, height: pane.height)
                    : (x: size.width / 2, y: 0.0, width: pane.width, height: pane.height))
        } else {
            drawDepth(
                slot, encoder: encoder, turns: turns,
                pane: (x: 0.0, y: 0.0, width: size.width, height: size.height), mode: Float(mode))
        }
        encoder.endEncoding()
        commands.addCompletedHandler { _ in pool.release(slot) }
        commands.present(drawable)
        commands.commit()
    }

    private func drawDepth(
        _ slot: ResultSlot, encoder: MTLRenderCommandEncoder, turns: Int,
        pane: (x: Double, y: Double, width: Double, height: Double), mode: Float
    ) {
        let aspect = pane.width / pane.height
        let modelAspect = Double(slot.camera.width) / Double(slot.camera.height)
        let displayAspect = turns.isMultiple(of: 2) ? modelAspect : 1 / modelAspect
        var uniforms = RenderUniforms(
            scaleNearFar: SIMD3(slot.metricScale, slot.depthRange.x, slot.depthRange.y),
            mode: mode,
            blend: blend,
            uvScale: SIMD2(
                Float(min(1, aspect / displayAspect)),
                Float(min(1, displayAspect / aspect))),
            quarterTurns: Int32(turns))
        encoder.setViewport(MTLViewport(
            originX: pane.x, originY: pane.y, width: pane.width, height: pane.height,
            znear: 0, zfar: 1))
        encoder.setRenderPipelineState(depthPipeline)
        encoder.setFragmentTexture(slot.camera, index: 0)
        encoder.setFragmentTexture(slot.inverseDepthTexture, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<RenderUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }

    private func drawARKit(
        _ slot: ResultSlot, encoder: MTLRenderCommandEncoder, turns: Int,
        pane: (x: Double, y: Double, width: Double, height: Double)
    ) {
        // Crop the 4:3 depth map to the model's field of view, so both panes line up.
        let arkit = slot.arkitDepth
        let fill = fillScale(
            from: (width: arkit.width, height: arkit.height),
            to: (width: slot.camera.width, height: slot.camera.height))
        let crop = SIMD2(
            Float(min(1, Double(slot.camera.width) / (fill * Double(arkit.width)))),
            Float(min(1, Double(slot.camera.height) / (fill * Double(arkit.height)))))
        let aspect = pane.width / pane.height
        let modelAspect = Double(slot.camera.width) / Double(slot.camera.height)
        let displayAspect = turns.isMultiple(of: 2) ? modelAspect : 1 / modelAspect
        var uniforms = MetersUniforms(
            nearFar: slot.depthRange,
            uvScale: SIMD2(
                Float(min(1, aspect / displayAspect)),
                Float(min(1, displayAspect / aspect))),
            cropScale: crop,
            cropOffset: (SIMD2<Float>(repeating: 1) - crop) / 2,
            quarterTurns: Int32(turns))
        encoder.setViewport(MTLViewport(
            originX: pane.x, originY: pane.y, width: pane.width, height: pane.height,
            znear: 0, zfar: 1))
        encoder.setRenderPipelineState(metersPipeline)
        encoder.setFragmentTexture(arkit, index: 0)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<MetersUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }
}
