import AVFoundation
import CoreVideo
import Foundation
import QuartzCore
import Synchronization

struct EngineInfo: Sendable {
    let preset: String
    let window: Int
    let resolutions: [Resolution]
    let neuralEngine: Bool

    var computeLabel: String { neuralEngine ? "Neural Engine · float16" : "GPU · float16" }
}

struct EngineStatus: Sendable {
    var processedFPS = 0.0
    var stepMs = 0.0
    var latencyMs = 0.0
    var focalPixels: Double?
    var depthRange: ClosedRange<Double>?
    var calibrating = true
    var switching: String?
    var error: String?
}

private struct CameraFrame: @unchecked Sendable {
    let generation: Int
    let image: CVPixelBuffer
    let captureTime: Double
    let focalPixels: Double
}

private final class Mailbox: Sendable {
    private let frame = Mutex<CameraFrame?>(nil)
    private let signal = DispatchSemaphore(value: 0)

    func putDroppingUnread(_ newFrame: CameraFrame) {
        let replaced = frame.withLock { slot in
            defer { slot = newFrame }
            return slot != nil
        }
        if !replaced { signal.signal() }
    }

    func take() -> CameraFrame {
        while true {
            signal.wait()
            if let next = frame.withLock({ slot in defer { slot = nil }; return slot }) {
                return next
            }
        }
    }
}

/// The camera session, with the focal length its format implies.
private final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private static let frameRate = 30.0
    private static let framesPerSecondTolerance = 0.1

    let generation: Int
    let position: AVCaptureDevice.Position
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "depthpro2.camera", qos: .userInteractive)
    private let onFrame: @Sendable (CVPixelBuffer, Double, Double) -> Void
    private let focalPixels: Double

    init(
        position: AVCaptureDevice.Position, generation: Int, covering width: Int, height: Int,
        onFrame: @escaping @Sendable (CVPixelBuffer, Double, Double) -> Void
    ) throws {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
            throw EngineError("This iPhone has no \(position == .front ? "front" : "back") camera.")
        }
        let format = try Self.closestFormat(of: device, coveringWidth: width, height: height)
        try device.lockForConfiguration()
        device.activeFormat = format
        let duration = CMTime(value: 1, timescale: Int32(Self.frameRate))
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
        device.videoZoomFactor = 1
        device.unlockForConfiguration()

        let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        focalPixels = Double(dimensions.width) / 2 / tan(Double(format.videoFieldOfView) * .pi / 360)

        self.position = position
        self.generation = generation
        self.onFrame = onFrame
        super.init()

        let input = try AVCaptureDeviceInput(device: device)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .inputPriority
        guard session.canAddInput(input), session.canAddOutput(output) else {
            throw EngineError("Cannot attach the camera to a capture session.")
        }
        session.addInput(input)
        session.addOutput(output)
        guard let connection = output.connection(with: .video) else {
            throw EngineError("The camera has no video connection.")
        }
        // The front sensor sits 180° from the back sensor; keep both buffers landscape.
        connection.videoRotationAngle = position == .front ? 180 : 0
        connection.automaticallyAdjustsVideoMirroring = false
        connection.isVideoMirrored = position == .front
    }

    func start() {
        queue.sync { session.startRunning() }
    }

    func stop() {
        queue.sync { if session.isRunning { session.stopRunning() } }
    }

    func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let image = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame(image, CACurrentMediaTime(), focalPixels)
    }

    /// The landscape format that covers the model and is closest to its aspect, then the smallest.
    private static func closestFormat(
        of device: AVCaptureDevice, coveringWidth width: Int, height: Int
    ) throws -> AVCaptureDevice.Format {
        let aspect = Double(width) / Double(height)
        let candidates = device.formats.compactMap { format -> (format: AVCaptureDevice.Format, mismatch: Double, area: Int)? in
            let size = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard Int(size.width) >= width, Int(size.height) >= height,
                  format.videoSupportedFrameRateRanges.contains(where: {
                      $0.maxFrameRate >= frameRate - framesPerSecondTolerance
                  }), format.videoFieldOfView > 0
            else { return nil }
            let mismatch = abs(Double(size.width) / Double(size.height) - aspect)
            return (format, mismatch, Int(size.width) * Int(size.height))
        }
        guard let best = candidates.min(by: { ($0.mismatch, $0.area) < ($1.mismatch, $1.area) }) else {
            throw EngineError("The camera has no landscape format covering \(width)×\(height).")
        }
        return best.format
    }
}

final class DepthEngine: @unchecked Sendable {
    let gpu: GPU
    var info: EngineInfo { infoBox.withLock { $0 } }
    private var cameraCover: (width: Int, height: Int)
    private let compiledBox: Mutex<CompiledModelPackage>
    private let infoBox: Mutex<EngineInfo>
    private let currentPackage: Mutex<ModelPackage>
    private let runnerBox: Mutex<ModelRunner?>
    private let currentResolution: Mutex<Resolution>
    private let status: Mutex<EngineStatus>
    private let recalibration = Mutex(false)
    private let processing = NSLock()
    private let mailbox = Mailbox()
    private let cameraBox = Mutex<Camera?>(nil)
    private let cameraGeneration = Atomic<Int>(0)
    private let nextGeneration = Atomic<Int>(1)
    private let frameHandler: Mutex<(@Sendable () -> Void)?>

    init(package: ModelPackage) throws {
        gpu = try GPU()
        let compiled = try CompiledModelPackage(packageURL: package.url)
        assert(package.singleFunction == compiled.singleFunction, "The package name and its functions disagree.")
        let (model, spec, resolutions) = try compiled.load(function: nil)
        assert(resolutions.allSatisfy { $0.width > $0.height }, "The model functions must be landscape.")
        guard let initial = resolutions.first(where: { $0.width == spec.width && $0.height == spec.height }),
              let cover = resolutions.max(by: { $0.width * $0.height < $1.width * $1.height })
        else { throw EngineError("The model package has no resolutions.") }
        cameraCover = (cover.width, cover.height)
        let runner = try ModelRunner(gpu: gpu, model: model, spec: spec)
        compiledBox = Mutex(compiled)
        infoBox = Mutex(
            EngineInfo(
                preset: spec.preset, window: spec.window, resolutions: resolutions,
                neuralEngine: compiled.singleFunction))
        currentPackage = Mutex(package)
        runnerBox = Mutex(runner)
        currentResolution = Mutex(initial)
        status = Mutex(EngineStatus())
        frameHandler = Mutex(nil)
        attach(runner)
    }

    /// Routes each published result to the view, including after a runner swap.
    func setOnFrame(_ handler: (@Sendable () -> Void)?) {
        frameHandler.withLock { $0 = handler }
    }

    private func attach(_ runner: ModelRunner) {
        runner.pool.onPublish = { [weak self] in
            self?.frameHandler.withLock { $0?() }
        }
    }

    func start() async throws {
        guard await Self.requestCameraAccess() else {
            throw EngineError("Camera access is off. Allow it in Settings → Depth Pro 2 → Camera.")
        }
        try await useCamera(position: .back, covering: cameraCover)
        let thread = Thread { [self] in runInference() }
        thread.name = "depthpro2.inference"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    var resolution: Resolution { currentResolution.withLock { $0 } }

    var currentPool: ResultPool? { runnerBox.withLock { $0?.pool } }

    func snapshot() -> EngineStatus { status.withLock { $0 } }

    func recalibrate() { recalibration.withLock { $0 = true } }

    /// Runs the model at `resolution`. The new model is built before the old one is dropped.
    func select(_ resolution: Resolution) async throws {
        guard resolution != self.resolution else { return }
        try beginSwitch("Loading \(resolution.title)…")
        defer { endSwitch() }
        let (compiled, gpu) = (compiledBox.withLock { $0 }, self.gpu)
        let runner = try await Task.detached(priority: .userInitiated) {
            let (model, spec, _) = try compiled.load(function: resolution.function)
            return try ModelRunner(gpu: gpu, model: model, spec: spec)
        }.value
        attach(runner)
        runnerBox.withLock { $0 = runner }
        currentResolution.withLock { $0 = resolution }
        status.withLock {
            $0.processedFPS = 0
            $0.stepMs = 0
            $0.error = nil
            $0.calibrating = true
        }
    }

    /// Switches to another package, at the matching resolution. The new model is built before the old is dropped.
    func selectModel(_ package: ModelPackage) async throws {
        guard package.name != currentPackage.withLock({ $0.name }) else { return }
        try beginSwitch("Loading \(package.label)…")
        defer { endSwitch() }
        let (resolution, gpu) = (self.resolution, self.gpu)
        let (compiled, runner, spec, resolutions) = try await Task.detached(priority: .userInitiated) {
            let compiled = try CompiledModelPackage(packageURL: package.url)
            let function = package.singleFunction ? nil : resolution.function
            let (model, spec, resolutions) = try compiled.load(function: function)
            let runner = try ModelRunner(gpu: gpu, model: model, spec: spec)
            return (compiled, runner, spec, resolutions)
        }.value
        assert(package.singleFunction == compiled.singleFunction, "The package name and its functions disagree.")
        guard let next = resolutions.first(where: { $0.width == spec.width && $0.height == spec.height }),
              let cover = resolutions.max(by: { $0.width * $0.height < $1.width * $1.height })
        else { throw EngineError("The model package has no resolutions.") }
        if cover.width * cover.height > cameraCover.width * cameraCover.height {
            // Frame the new package's largest resolution without upscaling the camera feed.
            try await useCamera(position: cameraPosition, covering: (cover.width, cover.height))
            cameraCover = (cover.width, cover.height)
        }
        attach(runner)
        compiledBox.withLock { $0 = compiled }
        runnerBox.withLock { $0 = runner }
        infoBox.withLock {
            $0 = EngineInfo(
                preset: spec.preset, window: spec.window, resolutions: resolutions,
                neuralEngine: compiled.singleFunction)
        }
        currentPackage.withLock { $0 = package }
        currentResolution.withLock { $0 = next }
        status.withLock {
            $0.processedFPS = 0
            $0.stepMs = 0
            $0.error = nil
            $0.calibrating = true
        }
    }

    func flipCamera() async throws {
        let position: AVCaptureDevice.Position = cameraPosition == .front ? .back : .front
        try beginSwitch("Switching camera…")
        defer { endSwitch() }
        try await useCamera(position: position, covering: cameraCover)
    }

    /// Claims the one model/camera switch slot, so two loads cannot swap runners concurrently.
    private func beginSwitch(_ message: String) throws {
        let busy = status.withLock { status in
            guard status.switching == nil else { return true }
            status.switching = message
            return false
        }
        guard !busy else { throw EngineError("Another model is loading.") }
    }

    private func endSwitch() {
        status.withLock { $0.switching = nil }
    }

    private var cameraPosition: AVCaptureDevice.Position {
        cameraBox.withLock { $0?.position ?? .back }
    }

    private func useCamera(
        position: AVCaptureDevice.Position, covering cover: (width: Int, height: Int)
    ) async throws {
        let generation = nextGeneration.add(1, ordering: .relaxed).newValue
        let camera = try Camera(
            position: position, generation: generation, covering: cover.width, height: cover.height,
            onFrame: { [weak self] image, time, focal in
                self?.mailbox.putDroppingUnread(
                    CameraFrame(
                        generation: generation, image: image, captureTime: time, focalPixels: focal))
            })
        cameraBox.withLock { $0 }?.stop()
        camera.start()
        cameraBox.withLock { $0 = camera }
        cameraGeneration.store(generation, ordering: .releasing)
        status.withLock {
            $0.focalPixels = nil
            $0.depthRange = nil
            $0.calibrating = true
        }
    }

    private static func requestCameraAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    private func runInference() {
        var runnerID: Int?
        var seq = 0
        var generation = 0
        var warmup = 0
        var smoother = DepthRangeSmoother()
        var processedTimes: [Double] = []
        var stepMs = 0.0
        var latencyMs = 0.0
        while true {
            let frame = mailbox.take()
            processing.lock()
            defer { processing.unlock() }
            let currentGeneration = cameraGeneration.load(ordering: .acquiring)
            guard let runner = runnerBox.withLock({ $0 }), frame.generation == currentGeneration else { continue }
            if runner.id != runnerID {
                runnerID = runner.id
                generation = 0
                warmup = 0
                smoother = DepthRangeSmoother()
                processedTimes.removeAll()
                stepMs = 0
            }
            if frame.generation != generation {
                runner.stream.reset()
                smoother = DepthRangeSmoother()
                generation = frame.generation
                warmup = 0
                processedTimes.removeAll()
            }
            if takeRecalibrationRequest() {
                smoother = DepthRangeSmoother()
            }
            do {
                let slot = runner.pool.acquireWritable()
                let times = try runner.process(frame.image, into: slot)
                warmup += 1
                guard warmup >= runner.spec.window else { continue }
                let now = CACurrentMediaTime()
                let scale = metricScale(focalPixels: frame.focalPixels, frame: frame.image, spec: runner.spec)
                if let measured = measuredDepthRange(in: slot.inverseDepth, scale: scale) {
                    smoother.update(with: measured, at: frame.captureTime)
                }
                guard let range = smoother.range else {
                    status.withLock { $0.error = "No valid depth in this frame." }
                    continue
                }
                slot.seq = seq
                slot.captureTime = frame.captureTime
                slot.publishTime = now
                slot.stepMs = times.preprocessMs + times.stepMs
                slot.metricScale = Float(scale)
                slot.depthRange = SIMD2(Float(range.lowerBound), Float(range.upperBound))
                runner.pool.publish(slot)

                processedTimes.append(now)
                processedTimes.removeAll { $0 < now - 2 }
                stepMs = stepMs == 0 ? slot.stepMs : stepMs + 0.15 * (slot.stepMs - stepMs)
                latencyMs = latencyMs == 0 ? 1000 * (now - frame.captureTime) : latencyMs + 0.15 * (1000 * (now - frame.captureTime) - latencyMs)
                let fps = Self.rate(processedTimes)
                status.withLock {
                    $0.processedFPS = fps
                    $0.stepMs = stepMs
                    $0.latencyMs = latencyMs
                    $0.focalPixels = frame.focalPixels
                    $0.depthRange = range
                    $0.calibrating = false
                    $0.error = nil
                }
                seq += 1
            } catch {
                status.withLock { $0.error = error.localizedDescription }
            }
        }
    }

    private func takeRecalibrationRequest() -> Bool {
        recalibration.withLock { requested in
            defer { requested = false }
            return requested
        }
    }

    private static func rate(_ times: [Double]) -> Double {
        guard let first = times.first, let last = times.last, last > first else { return 0 }
        return Double(times.count - 1) / (last - first)
    }
}
