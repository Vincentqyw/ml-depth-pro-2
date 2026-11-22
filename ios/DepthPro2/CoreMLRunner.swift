import CoreML
import CoreVideo
import Foundation
import Metal
import QuartzCore
import Synchronization

struct EngineError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

struct Resolution: Identifiable, Hashable, Sendable {
    let function: String
    let width: Int
    let height: Int

    var id: String { function }
    var title: String { "\(width)×\(height)" }

    /// The functions of a package, named `res_<width>x<height>`, smallest first.
    static func list(_ names: [String]) -> [Resolution] {
        names.compactMap { name in
            guard name.hasPrefix("res_") else { return nil }
            let parts = name.dropFirst(4).split(separator: "x").compactMap { Int($0) }
            guard parts.count == 2 else { return nil }
            return Resolution(function: name, width: parts[0], height: parts[1])
        }
        .sorted { ($0.width * $0.height, $0.width) < ($1.width * $1.height, $1.width) }
    }
}

struct ModelSpec: Sendable {
    let preset: String
    let window: Int
    let height: Int
    let width: Int

    var label: String { "\(preset.uppercased())" }
    var detail: String { "\(width)×\(height) · W\(window)" }
}

/// A depth package in the app bundle, with the label the model picker shows.
struct ModelPackage: Identifiable, Hashable, Sendable {
    let name: String
    let preset: String
    let window: Int
    /// Exported as `<preset>_<H>x<W>_w<window>` for one size, instead of `<preset>_w<window>`.
    let singleFunction: Bool
    let label: String
    let note: String
    let url: URL

    var id: String { name }

    /// The bundled `depthpro2-*.mlpackage` packages, fastest preset and window first.
    static func bundled() -> [ModelPackage] {
        guard let directory = Bundle.main.resourceURL?.appendingPathComponent("Models") else { return [] }
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)) ?? []
        let notes = ["S": "Fastest", "B": "Sharper, slower", "L": "Sharpest, slowest", "H": "Highest quality"]
        let order = ["S": 0, "B": 1, "L": 2, "H": 3]
        return urls.compactMap { url -> ModelPackage? in
            guard url.pathExtension == "mlpackage" else { return nil }
            let name = url.deletingPathExtension().lastPathComponent
            let parts = name.split(separator: "_")
            guard parts.count == 2 || parts.count == 3,
                  let suffix = parts.last, suffix.hasPrefix("w"), let window = Int(suffix.dropFirst()),
                  let preset = parts.first?.split(separator: "-").last.map({ $0.uppercased() })
            else { return nil }
            let singleFunction = parts.count == 3
            let size = singleFunction ? parts[1].split(separator: "x").compactMap { Int($0) } : []
            assert(size.isEmpty || size.count == 2, "A single-function package name must state its HxW.")
            return ModelPackage(
                name: name, preset: preset, window: window, singleFunction: singleFunction,
                label: "Depth Pro 2 \(preset) · W\(window)"
                    + (size.count == 2 ? " · \(size[1])×\(size[0])" : ""),
                note: singleFunction ? "Neural Engine · fastest"
                    : window == 4 ? notes[preset] ?? "Higher quality" : "GPU · faster, less stable",
                url: url)
        }
        .sorted {
            (order[$0.preset] ?? 9, -$0.window, $0.singleFunction ? 0 : 1, $0.name)
                < (order[$1.preset] ?? 9, -$1.window, $1.singleFunction ? 0 : 1, $1.name)
        }
    }
}

/// A package compiled once, so each function loads without compiling again.
final class CompiledModelPackage: @unchecked Sendable {
    let url: URL
    /// A package with a single function (exported with one size) runs on the Neural Engine;
    /// iOS 26 rejects multifunction packages there, so those run on the GPU.
    let singleFunction: Bool

    init(packageURL: URL) throws {
        url = try MLModel.compileModel(at: packageURL)
        // Multifunction packages declare their functions in the model spec's metadata.
        let spec = try Data(
            contentsOf: packageURL.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel"))
        singleFunction = spec.range(of: Data("depthpro2.functions".utf8)) == nil
    }

    /// `function` of the package (nil: its default), with its spec and the package's resolutions.
    func load(function: String?) throws -> (model: MLModel, spec: ModelSpec, resolutions: [Resolution]) {
        if singleFunction, let function, !function.isEmpty {
            throw EngineError("The single-function package cannot load \(function).")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = singleFunction ? .cpuAndNeuralEngine : .cpuAndGPU
        configuration.allowLowPrecisionAccumulationOnGPU = true
        if let function, !function.isEmpty { configuration.functionName = function }
        let model = try MLModel(contentsOf: url, configuration: configuration)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard let preset = metadata["demo.preset"], let window = metadata["demo.window"].flatMap(Int.init) else {
            throw EngineError("The model package is missing its Depth Pro 2 metadata.")
        }
        guard let shape = model.modelDescription.inputDescriptionsByName["frame"]?.multiArrayConstraint?.shape,
              shape.count == 4
        else { throw EngineError("The model package has no frame input of shape (1, 3, H, W).") }
        guard model.modelDescription.outputDescriptionsByName["inverse_depth"] != nil else {
            throw EngineError("The model package has no inverse_depth output.")
        }
        let spec = ModelSpec(
            preset: preset, window: window, height: shape[2].intValue, width: shape[3].intValue)
        let names = metadata["depthpro2.functions"]?.split(separator: ",").map(String.init) ?? []
        let resolutions = names.isEmpty
            ? [Resolution(function: "", width: spec.width, height: spec.height)]
            : Resolution.list(names)
        guard !resolutions.isEmpty else {
            throw EngineError("The model package lists no resolutions.")
        }
        return (model, spec, resolutions)
    }
}

final class DepthStream {
    let spec: ModelSpec
    let input: CVPixelBuffer
    private let model: MLModel
    private let features: MLFeatureProvider
    private var state: MLState?

    init(model: MLModel, spec: ModelSpec) throws {
        self.model = model
        self.spec = spec
        input = try makeHalfSurface(width: spec.width, height: 3 * spec.height)
        let frame = MLMultiArray(
            pixelBuffer: input, shape: [1, 3, spec.height, spec.width].map { NSNumber(value: $0) })
        features = try MLDictionaryFeatureProvider(dictionary: ["frame": MLFeatureValue(multiArray: frame)])
        reset()
    }

    /// Forgets the earlier frames, so the next step starts a new window.
    func reset() {
        state = spec.window > 1 ? model.makeState() : nil
    }

    func step(predictionOptions: MLPredictionOptions) throws {
        if let state {
            _ = try model.prediction(from: features, using: state, options: predictionOptions)
        } else {
            _ = try model.prediction(from: features, options: predictionOptions)
        }
    }
}

final class ResultSlot: @unchecked Sendable {
    let camera: MTLTexture
    let inverseDepth: CVPixelBuffer
    let inverseDepthTexture: MTLTexture
    /// The LiDAR depth ARKit measured for the same frame, in meters (256×192 on LiDAR iPhones).
    let arkitDepth: MTLTexture
    let predictionOptions = MLPredictionOptions()
    var seq = -1
    var captureTime = 0.0
    var publishTime = 0.0
    var stepMs = 0.0
    var metricScale: Float = 0
    var depthRange = SIMD2<Float>.zero
    /// The median absolute difference from ARKit, in meters.
    var arkitError: Float?
    fileprivate var readers = 0

    init(gpu: GPU, spec: ModelSpec) throws {
        camera = try gpu.texture(.bgra8Unorm, width: spec.width, height: spec.height)
        inverseDepth = try makeHalfSurface(width: spec.width, height: spec.height)
        inverseDepthTexture = try gpu.texture(inverseDepth, format: .r16Float, usage: .shaderRead)
        // Shared storage, because the LiDAR map is copied in from the CPU.
        arkitDepth = try gpu.texture(.r32Float, width: 256, height: 192, storage: .shared)
        let backing = MLMultiArray(
            pixelBuffer: inverseDepth, shape: [1, spec.height, spec.width].map { NSNumber(value: $0) })
        predictionOptions.outputBackings = ["inverse_depth": backing]
    }

    func copyARKitDepth(_ depth: CVPixelBuffer) throws {
        assert(CVPixelBufferGetPixelFormatType(depth) == kCVPixelFormatType_DepthFloat32)
        assert(
            CVPixelBufferGetWidth(depth) == arkitDepth.width
                && CVPixelBufferGetHeight(depth) == arkitDepth.height,
            "ARKit changed its depth map size.")
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(depth) else {
            throw EngineError("The ARKit depth map has no base address.")
        }
        arkitDepth.replace(
            region: MTLRegionMake2D(0, 0, arkitDepth.width, arkitDepth.height), mipmapLevel: 0,
            withBytes: base, bytesPerRow: CVPixelBufferGetBytesPerRow(depth))
    }
}

final class ResultPool: @unchecked Sendable {
    private let slots: [ResultSlot]
    private let condition = NSCondition()
    private var latest: ResultSlot?
    var onPublish: (@Sendable () -> Void)?

    init(gpu: GPU, spec: ModelSpec, count: Int) throws {
        slots = try (0..<count).map { _ in try ResultSlot(gpu: gpu, spec: spec) }
    }

    func acquireWritable() -> ResultSlot {
        condition.lock()
        defer { condition.unlock() }
        while true {
            if let slot = slots.first(where: { $0 !== latest && $0.readers == 0 }) { return slot }
            condition.wait()
        }
    }

    func publish(_ slot: ResultSlot) {
        condition.lock()
        latest = slot
        condition.unlock()
        onPublish?()
    }

    func acquireLatest() -> ResultSlot? {
        condition.lock()
        defer { condition.unlock() }
        latest?.readers += 1
        return latest
    }

    func release(_ slot: ResultSlot) {
        condition.lock()
        slot.readers -= 1
        condition.signal()
        condition.unlock()
    }
}

private struct ResizeUniforms {
    var sourceSize: SIMD2<Float>
    var scale: SIMD2<Float>
    var translate: SIMD2<Float>
}

final class Preprocess {
    private let bgraPipeline: MTLComputePipelineState
    private let yuvPipeline: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache
    private let gpu: GPU

    init(gpu: GPU) throws {
        self.gpu = gpu
        guard let bgra = gpu.library.makeFunction(name: "resize_split"),
              let yuv = gpu.library.makeFunction(name: "resize_split_yuv")
        else { throw EngineError("The preprocess shaders are missing.") }
        bgraPipeline = try gpu.device.makeComputePipelineState(function: bgra)
        yuvPipeline = try gpu.device.makeComputePipelineState(function: yuv)
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, gpu.device, nil, &cache)
        guard let cache else { throw EngineError("Cannot create a Metal texture cache.") }
        textureCache = cache
    }

    func run(_ image: CVPixelBuffer, camera: MTLTexture, planar: MTLTexture) throws {
        let source = (width: CVPixelBufferGetWidth(image), height: CVPixelBufferGetHeight(image))
        let yuv = CVPixelBufferGetPixelFormatType(image) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        var wrapped: [CVMetalTexture] = []
        func plane(_ index: Int, _ format: MTLPixelFormat, _ width: Int, _ height: Int) throws -> MTLTexture {
            var cvTexture: CVMetalTexture?
            CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, textureCache, image, nil, format, width, height, index, &cvTexture)
            guard let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else {
                throw EngineError("Cannot wrap the camera frame as a texture.")
            }
            wrapped.append(cvTexture)
            return texture
        }
        let luma = yuv ? try plane(0, .r8Unorm, source.width, source.height) : nil
        let chroma = yuv ? try plane(1, .rg8Unorm, source.width / 2, source.height / 2) : nil
        let bgra = yuv ? nil : try plane(0, .bgra8Unorm, source.width, source.height)

        let (width, height) = (Double(camera.width), Double(camera.height))
        let scale = fillScale(from: source, to: (width: camera.width, height: camera.height))
        var uniforms = ResizeUniforms(
            sourceSize: SIMD2(Float(source.width), Float(source.height)),
            scale: SIMD2(Float(scale), Float(scale)),
            translate: SIMD2(
                Float((width - Double(source.width) * scale) / 2),
                Float((height - Double(source.height) * scale) / 2)))

        guard let commands = gpu.queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder()
        else { throw EngineError("Cannot make a compute encoder.") }
        if let luma, let chroma {
            encoder.setComputePipelineState(yuvPipeline)
            encoder.setTexture(luma, index: 0)
            encoder.setTexture(chroma, index: 1)
        } else if let bgra {
            encoder.setComputePipelineState(bgraPipeline)
            encoder.setTexture(bgra, index: 0)
        } else {
            throw EngineError("Cannot wrap the camera frame as a texture.")
        }
        encoder.setTexture(camera, index: 2)
        encoder.setTexture(planar, index: 3)
        encoder.setBytes(&uniforms, length: MemoryLayout<ResizeUniforms>.stride, index: 0)
        encoder.dispatchThreads(
            MTLSize(width: camera.width, height: camera.height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        encoder.endEncoding()
        withExtendedLifetime(wrapped) {
            commands.commit()
            commands.waitUntilCompleted()
        }
        CVMetalTextureCacheFlush(textureCache, 0)
        if let error = commands.error { throw error }
    }
}

final class ModelRunner: @unchecked Sendable {
    private static let slotCount = 4
    private static let count = Atomic<Int>(0)

    let id = count.add(1, ordering: .relaxed).newValue
    let spec: ModelSpec
    let stream: DepthStream
    let pool: ResultPool
    private let planar: MTLTexture
    private let preprocess: Preprocess

    init(gpu: GPU, model: MLModel, spec: ModelSpec) throws {
        self.spec = spec
        stream = try DepthStream(model: model, spec: spec)
        // The model reads `stream.input`: the planar planes must be written into that same buffer.
        planar = try gpu.texture(stream.input, format: .r16Float, usage: .shaderWrite)
        preprocess = try Preprocess(gpu: gpu)
        pool = try ResultPool(gpu: gpu, spec: spec, count: Self.slotCount)
    }

    /// Runs the model on `image` into `slot` and returns how long processing and inference took.
    func process(
        _ image: CVPixelBuffer, sceneDepth: CVPixelBuffer?, into slot: ResultSlot
    ) throws -> (preprocessMs: Double, stepMs: Double) {
        let preprocessStart = CACurrentMediaTime()
        try preprocess.run(image, camera: slot.camera, planar: planar)
        if let sceneDepth {
            try slot.copyARKitDepth(sceneDepth)
        }
        let stepStart = CACurrentMediaTime()
        try stream.step(predictionOptions: slot.predictionOptions)
        let end = CACurrentMediaTime()
        return (1000 * (stepStart - preprocessStart), 1000 * (end - stepStart))
    }
}

/// A disk of the model's input image, in its pixels.
struct DepthRegion: Sendable, Equatable {
    var center: SIMD2<Float>
    var radius: Float

    func contains(_ x: Int, _ y: Int) -> Bool {
        let dx = Float(x) + 0.5 - center.x
        let dy = Float(y) + 0.5 - center.y
        return dx * dx + dy * dy <= radius * radius
    }
}

/// Depth Pro 2 and ARKit's depth over a region of the image, in meters.
struct RegionStats: Sendable, Equatable {
    let model: Float
    let arkit: Float
    let error: Float
}

/// The median absolute difference between the model's depth and ARKit's, in meters, over the
/// pixels where ARKit is confident and both estimates are valid, and the same statistics within
/// `region`, if it has enough samples.
func depthComparison(
    inverseDepth: CVPixelBuffer, sceneDepth: CVPixelBuffer, image: CVPixelBuffer,
    confidence: CVPixelBuffer?, scale: Double, spec: ModelSpec, region: DepthRegion?
) -> (error: Float?, region: RegionStats?) {
    let arkit = (width: CVPixelBufferGetWidth(sceneDepth), height: CVPixelBufferGetHeight(sceneDepth))
    // The map covers the camera image's field of view, which is what makes sampling it at the same
    // normalized coordinates correct.
    assert(
        abs(Double(arkit.width) / Double(arkit.height)
            - Double(CVPixelBufferGetWidth(image)) / Double(CVPixelBufferGetHeight(image))) < 0.01,
        "ARKit's depth map and camera image have different fields of view.")
    let fill = fillScale(from: arkit, to: (width: spec.width, height: spec.height))
    let offset = (
        x: (Double(spec.width) - Double(arkit.width) * fill) / 2,
        y: (Double(spec.height) - Double(arkit.height) * fill) / 2)
    CVPixelBufferLockBaseAddress(inverseDepth, .readOnly)
    CVPixelBufferLockBaseAddress(sceneDepth, .readOnly)
    if let confidence { CVPixelBufferLockBaseAddress(confidence, .readOnly) }
    defer {
        CVPixelBufferUnlockBaseAddress(inverseDepth, .readOnly)
        CVPixelBufferUnlockBaseAddress(sceneDepth, .readOnly)
        if let confidence { CVPixelBufferUnlockBaseAddress(confidence, .readOnly) }
    }
    guard let modelBase = CVPixelBufferGetBaseAddress(inverseDepth),
          let depthBase = CVPixelBufferGetBaseAddress(sceneDepth)
    else { return (nil, nil) }
    let (modelRowBytes, depthRowBytes) = (
        CVPixelBufferGetBytesPerRow(inverseDepth), CVPixelBufferGetBytesPerRow(sceneDepth))
    let confidenceBase = confidence.flatMap { CVPixelBufferGetBaseAddress($0) }
    let confidenceRowBytes = confidence.map { CVPixelBufferGetBytesPerRow($0) } ?? 0
    var errors: [Float] = []
    var regionErrors: [Float] = []
    var regionModel: [Float] = []
    var regionARKit: [Float] = []
    let step = 2
    for y in stride(from: 0, to: arkit.height, by: step) {
        let depthRow = (depthBase + y * depthRowBytes).assumingMemoryBound(to: Float.self)
        let confidenceRow = confidenceBase.map {
            ($0 + y * confidenceRowBytes).assumingMemoryBound(to: UInt8.self)
        }
        let modelY = Int(offset.y + (Double(y) + 0.5) * fill)
        guard modelY >= 0, modelY < spec.height else { continue }
        let modelRow = (modelBase + modelY * modelRowBytes).assumingMemoryBound(to: Float16.self)
        for x in stride(from: 0, to: arkit.width, by: step) {
            if let confidenceRow, confidenceRow[x] < 2 { continue }
            let measured = depthRow[x]
            guard measured.isFinite, measured > 0.2, measured < 5 else { continue }
            let modelX = Int(offset.x + (Double(x) + 0.5) * fill)
            guard modelX >= 0, modelX < spec.width else { continue }
            let inverse = modelRow[modelX]
            guard inverse.isFinite, inverse > 0 else { continue }
            let model = Float(scale / Double(inverse))
            let difference = abs(model - measured)
            errors.append(difference)
            if region?.contains(modelX, modelY) == true {
                regionModel.append(model)
                regionARKit.append(measured)
                regionErrors.append(difference)
            }
        }
    }
    let error = errors.count >= 100 ? median(errors) : nil
    guard regionModel.count >= 10 else { return (error, nil) }
    return (
        error,
        RegionStats(
            model: median(regionModel), arkit: median(regionARKit), error: median(regionErrors)))
}

/// The middle of a non-empty sample.
private func median(_ values: [Float]) -> Float {
    assert(!values.isEmpty, "The median needs a sample.")
    return values.sorted()[values.count / 2]
}

func metricScale(focalPixels: Double, frame: CVPixelBuffer, spec: ModelSpec) -> Double {
    let source = (width: CVPixelBufferGetWidth(frame), height: CVPixelBufferGetHeight(frame))
    let target = (width: spec.width, height: spec.height)
    let focalInModelPixels = focalPixels * fillScale(from: source, to: target)
    return focalInModelPixels / Double(spec.width)
}

func measuredDepthRange(in inverseDepth: CVPixelBuffer, scale: Double) -> ClosedRange<Double>? {
    let (step, lowQuantile, highQuantile) = (8, 0.05, 0.95)
    CVPixelBufferLockBaseAddress(inverseDepth, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(inverseDepth, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(inverseDepth) else { return nil }
    let rowBytes = CVPixelBufferGetBytesPerRow(inverseDepth)
    var meters: [Double] = []
    for y in stride(from: 0, to: CVPixelBufferGetHeight(inverseDepth), by: step) {
        let row = (base + y * rowBytes).assumingMemoryBound(to: Float16.self)
        for x in stride(from: 0, to: CVPixelBufferGetWidth(inverseDepth), by: step)
        where row[x].isFinite && row[x] > 0 {
            meters.append(scale / Double(row[x]))
        }
    }
    guard !meters.isEmpty else { return nil }
    meters.sort()
    let last = Double(meters.count - 1)
    return meters[Int(lowQuantile * last)]...meters[Int(highQuantile * last)]
}

struct DepthRangeSmoother {
    private static let timeConstant = 0.5
    private static let longestStep = 0.1

    private var smoothed: SIMD2<Double>?
    private var time = 0.0

    var range: ClosedRange<Double>? { smoothed.map { exp($0.x)...exp($0.y) } }

    mutating func update(with measured: ClosedRange<Double>, at newTime: Double) {
        let target = SIMD2(log(measured.lowerBound), log(measured.upperBound))
        let previous = smoothed ?? target
        let step = min(max(0, newTime - time), Self.longestStep)
        let weight = smoothed == nil ? 1 : 1 - exp(-step / Self.timeConstant)
        smoothed = previous + weight * (target - previous)
        time = newTime
    }
}
