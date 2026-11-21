import MetalKit
import SwiftUI
import UIKit

struct MetalView: UIViewRepresentable {
    let engine: DepthEngine
    let mode: Int
    let blend: Float

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: engine.gpu.device)
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = false
        view.isPaused = false
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
        context.coordinator.renderer = renderer
        view.delegate = renderer
        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {
        context.coordinator.renderer.mode = mode
        context.coordinator.renderer.blend = blend
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

private extension UIInterfaceOrientation {
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
    private let pipeline: MTLRenderPipelineState
    var mode = 2
    var blend: Float = 0.8

    init(gpu: GPU, engine: DepthEngine) throws {
        self.gpu = gpu
        self.engine = engine
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = gpu.library.makeFunction(name: "fullscreen")
        descriptor.fragmentFunction = gpu.library.makeFunction(name: "display")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipeline = try gpu.device.makeRenderPipelineState(descriptor: descriptor)
        super.init()
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
        let screenAspect = Double(drawable.texture.width) / Double(drawable.texture.height)
        let modelAspect = Double(slot.camera.width) / Double(slot.camera.height)
        let displayAspect = turns.isMultiple(of: 2) ? modelAspect : 1 / modelAspect
        var uniforms = RenderUniforms(
            scaleNearFar: SIMD3(slot.metricScale, slot.depthRange.x, slot.depthRange.y),
            mode: Float(mode),
            blend: blend,
            uvScale: SIMD2(
                Float(min(1, screenAspect / displayAspect)),
                Float(min(1, displayAspect / screenAspect))),
            quarterTurns: Int32(turns))
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(slot.camera, index: 0)
        encoder.setFragmentTexture(slot.inverseDepthTexture, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<RenderUniforms>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commands.addCompletedHandler { _ in pool.release(slot) }
        commands.present(drawable)
        commands.commit()
    }
}
