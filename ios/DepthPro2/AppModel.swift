import Observation
import SwiftUI
import UIKit

enum ViewMode: Int, CaseIterable, Identifiable {
    case camera, blend, depth

    var id: Int { rawValue }

    var icon: String {
        switch self {
        case .camera: "camera.fill"
        case .blend: "circle.lefthalf.filled"
        case .depth: "square.stack.3d.up.fill"
        }
    }

    var label: String {
        switch self {
        case .camera: "Camera"
        case .blend: "Blend"
        case .depth: "Depth"
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

    var phase: Phase = .loading("Preparing the on-device model…")
    var mode: ViewMode = .blend
    var blend = 0.78
    var status = EngineStatus()
    var currentResolution: Resolution?
    var models: [ModelPackage] = []
    var selectedModel: ModelPackage?
    var showInfo = false
    var alert: String?
    private(set) var engine: DepthEngine?
    private var started = false
    @ObservationIgnored private var pollTask: Task<Void, Never>?

    var info: EngineInfo? { engine?.info }
    var resolutions: [Resolution] { engine?.info.resolutions ?? [] }

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

    func select(_ resolution: Resolution) {
        guard let engine, resolution != currentResolution, status.switching == nil else { return }
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
