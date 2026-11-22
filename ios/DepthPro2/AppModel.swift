import Observation
import SwiftUI
import UIKit

enum ViewMode: Int, CaseIterable, Identifiable {
    case camera, blend, depth, compare

    var id: Int { rawValue }

    var icon: String {
        switch self {
        case .camera: "camera.fill"
        case .blend: "circle.lefthalf.filled"
        case .depth: "square.stack.3d.up.fill"
        case .compare: "rectangle.split.2x1"
        }
    }

    var label: String {
        switch self {
        case .camera: "Camera"
        case .blend: "Blend"
        case .depth: "Depth"
        case .compare: "Compare"
        }
    }
}

@MainActor
@Observable
final class AppModel {
    enum Phase: Equatable {
        case loading(String)
        case ready
        case failed(String)
    }

    private static let modelKey = "depthpro2.model"
    /// The compare loupe's radius on screen, in points.
    static let regionRadius: CGFloat = 44

    var phase: Phase = .loading("Preparing the on-device model…")
    var mode: ViewMode = .blend
    var blend = 0.78
    var status = EngineStatus()
    var currentResolution: Resolution?
    var models: [ModelPackage] = []
    var selectedModel: ModelPackage?
    var showInfo = false
    var alert: String?
    /// The compare loupe's position on screen, in points.
    private(set) var regionPoint: CGPoint?
    /// The same disk's place in the other pane, in points.
    private(set) var regionMirrorPoint: CGPoint?
    private(set) var engine: DepthEngine?
    private var started = false
    @ObservationIgnored private var pollTask: Task<Void, Never>?

    var info: EngineInfo? { engine?.info }
    var resolutions: [Resolution] { engine?.info.resolutions ?? [] }
    var availableModes: [ViewMode] {
        DepthEngine.sceneDepthSupported
            ? ViewMode.allCases : ViewMode.allCases.filter { $0 != .compare }
    }

    func start() {
        guard !started, engine == nil else { return }
        let packages = ModelPackage.bundled()
        guard let fallback = packages.first else {
            phase = .failed("The app bundle has no Depth Pro 2 package in Models/.")
            return
        }
        models = packages
        let saved = UserDefaults.standard.string(forKey: Self.modelKey)
        let package = packages.first { $0.name == saved } ?? fallback
        selectedModel = package
        started = true
        phase = .loading("Preparing the on-device model…")
        let model = self
        Task.detached(priority: .userInitiated) {
            do {
                let engine = try DepthEngine(package: package)
                await model.setPhase(.loading("Starting the camera…"))
                try await engine.start()
                await model.ready(engine)
            } catch {
                await model.failed(error.localizedDescription)
            }
        }
    }

    private func setPhase(_ phase: Phase) { self.phase = phase }

    private func ready(_ engine: DepthEngine) {
        self.engine = engine
        currentResolution = engine.resolution
        status = engine.snapshot()
        phase = .ready
        poll()
    }

    private func failed(_ message: String) {
        started = false
        phase = .failed(message)
    }

    func retry() {
        engine = nil
        started = false
        start()
    }

    /// Compare mode runs ARKit instead of the capture session, so switching it swaps cameras.
    func setMode(_ newMode: ViewMode) {
        guard newMode != mode, status.switching == nil else { return }
        let previous = mode
        mode = newMode
        if previous == .compare { clearRegion() }
        guard let engine, (newMode == .compare) != (previous == .compare) else { return }
        Task {
            do {
                try await engine.setCompare(newMode == .compare)
            } catch {
                mode = previous
                alert = error.localizedDescription
            }
        }
    }

    /// Points the compare loupe at the image under `point`. Both panes show the model's field of
    /// view, so the touch's place within either pane names the same disk of the model image.
    func selectRegion(at point: CGPoint, in size: CGSize) {
        guard mode == .compare, status.switching == nil, let engine,
              let resolution = currentResolution, size.width > 0, size.height > 0
        else { return }
        let turns = Self.interfaceTurns
        let stacked = !turns.isMultiple(of: 2)
        let pane = CGSize(width: stacked ? size.width : size.width / 2,
                          height: stacked ? size.height / 2 : size.height)
        let local = CGPoint(x: (point.x.truncatingRemainder(dividingBy: pane.width)) / pane.width,
                            y: (point.y.truncatingRemainder(dividingBy: pane.height)) / pane.height)
        let aspect = pane.width / pane.height
        let modelAspect = CGFloat(resolution.width) / CGFloat(resolution.height)
        let displayAspect = turns.isMultiple(of: 2) ? modelAspect : 1 / modelAspect
        let uvScale = CGPoint(x: min(1, aspect / displayAspect), y: min(1, displayAspect / aspect))
        let display = CGPoint(x: (local.x - 0.5) * uvScale.x + 0.5,
                              y: (local.y - 0.5) * uvScale.y + 0.5)
        let uv: CGPoint = switch turns {
        case 1: CGPoint(x: display.y, y: 1 - display.x)
        case 2: CGPoint(x: 1 - display.x, y: 1 - display.y)
        case 3: CGPoint(x: 1 - display.y, y: display.x)
        default: display
        }
        let width = CGFloat(resolution.width)
        let height = CGFloat(resolution.height)
        let pixelsPerPoint = uvScale.x * (turns.isMultiple(of: 2) ? width : height) / pane.width
        let mirror = stacked
            ? CGPoint(x: 0, y: point.y < size.height / 2 ? pane.height : -pane.height)
            : CGPoint(x: point.x < size.width / 2 ? pane.width : -pane.width, y: 0)
        regionPoint = point
        regionMirrorPoint = CGPoint(x: point.x + mirror.x, y: point.y + mirror.y)
        engine.setRegion(
            DepthRegion(
                center: SIMD2(Float(uv.x * width), Float(uv.y * height)),
                radius: Float(Self.regionRadius * pixelsPerPoint)))
    }

    func clearRegion() {
        regionPoint = nil
        regionMirrorPoint = nil
        engine?.setRegion(nil)
    }

    func select(_ resolution: Resolution) {
        guard let engine, resolution != currentResolution, status.switching == nil else { return }
        clearRegion()
        Task {
            do {
                try await engine.select(resolution)
                currentResolution = engine.resolution
            } catch {
                alert = error.localizedDescription
            }
        }
    }

    func selectModel(_ package: ModelPackage) {
        guard let engine, package != selectedModel, status.switching == nil else { return }
        clearRegion()
        let previous = selectedModel
        selectedModel = package
        Task {
            do {
                try await engine.selectModel(package)
                UserDefaults.standard.set(package.name, forKey: Self.modelKey)
            } catch {
                selectedModel = previous
                alert = error.localizedDescription
            }
        }
    }

    func flipCamera() {
        guard let engine, status.switching == nil else { return }
        Task {
            do { try await engine.flipCamera() } catch { alert = error.localizedDescription }
        }
    }

    func recalibrate() { engine?.recalibrate() }

    func openSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }

    /// The renderer rotates the model into this interface; the compare loupe must do the same.
    private static var interfaceTurns: Int {
        guard let scene = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).first
        else { fatalError("The app runs in a window scene.") }
        return scene.interfaceOrientation.quarterTurns
    }

    private func poll() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(250))
                guard let self, let engine = self.engine else { continue }
                self.status = engine.snapshot()
                self.currentResolution = engine.resolution
            }
        }
    }
}
