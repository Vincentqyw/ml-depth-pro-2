import SwiftUI

struct ContentView: View {
    @State private var model = AppModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            switch model.phase {
            case .ready:
                if let engine = model.engine {
                    CameraScreen(model: model, engine: engine)
                        .transition(.opacity)
                }
            case .loading(let message):
                LaunchView(message: message)
                    .transition(.opacity)
            case .failed(let message):
                FailureView(message: message, retry: model.retry, openSettings: model.openSettings)
                    .transition(.opacity)
            }
        }
        .preferredColorScheme(.dark)
        .task { model.start() }
        .animation(.easeInOut(duration: 0.45), value: model.phase)
        .sensoryFeedback(.selection, trigger: model.mode)
    }
}

private struct CameraScreen: View {
    @Bindable var model: AppModel
    let engine: DepthEngine
    @Environment(\.verticalSizeClass) private var verticalSizeClass

    var body: some View {
        ZStack {
            MetalView(
                engine: engine, mode: model.mode.rawValue, blend: Float(model.blend),
                compare: model.mode == .compare)
                .ignoresSafeArea()

            if model.mode == .compare {
                CompareLabels(model: model, compactHeight: verticalSizeClass == .compact)
                    .allowsHitTesting(false)
            }

            LinearGradient(
                stops: [
                    .init(color: .black.opacity(0.52), location: 0),
                    .init(color: .clear, location: 0.22),
                    .init(color: .clear, location: 0.58),
                    .init(color: .black.opacity(0.72), location: 1),
                ],
                startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
                .allowsHitTesting(false)

            if model.mode == .compare {
                RegionSelection(model: model)
            }

            VStack(spacing: 10) {
                Header(model: model)
                Spacer(minLength: 0)
                StatusPills(model: model)
                if verticalSizeClass == .compact {
                    HStack(alignment: .bottom, spacing: 16) {
                        DepthColumn(model: model)
                            .frame(maxWidth: 340)
                        Spacer(minLength: 0)
                        Controls(model: model)
                    }
                } else {
                    VStack(spacing: 10) {
                        DepthColumn(model: model)
                        Controls(model: model)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 4)
            .padding(.bottom, 6)
            .animation(.snappy(duration: 0.32), value: model.mode)
            .animation(.snappy(duration: 0.32), value: model.status.depthRange != nil)
        }
        .sheet(isPresented: $model.showInfo) { InfoSheet(model: model) }
        .alert(
            "Something went wrong",
            isPresented: Binding(
                get: { model.alert != nil },
                set: { if !$0 { model.alert = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.alert ?? "")
        }
    }
}

/// Labels the two panes the renderer draws: stacked in portrait, side by side in landscape.
private struct CompareLabels: View {
    let model: AppModel
    let compactHeight: Bool

    var body: some View {
        let layout = compactHeight
            ? AnyLayout(HStackLayout(spacing: 0)) : AnyLayout(VStackLayout(spacing: 0))
        layout {
            pane("DEPTH PRO 2", detail: model.currentResolution?.title ?? "")
            pane("ARKIT LIDAR", detail: "256×192")
        }
        .padding(.horizontal, 20)
        .padding(.top, 68)
        .padding(.bottom, 150)
    }

    private func pane(_ title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 6) {
                Text(title)
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .tracking(1.2)
                Text(detail)
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .glass(Capsule())
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

/// A draggable loupe over the compare panes: the pill reports the depths inside it.
private struct RegionSelection: View {
    @Bindable var model: AppModel

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .topLeading) {
                Color.clear
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { model.selectRegion(at: $0.location, in: geometry.size) })
                if let point = model.regionPoint {
                    RegionLoupe()
                        .position(point)
                        .allowsHitTesting(false)
                }
                if let point = model.regionMirrorPoint {
                    RegionLoupe(mirrored: true)
                        .position(point)
                        .allowsHitTesting(false)
                }
            }
            .onChange(of: geometry.size) { _, _ in model.clearRegion() }
        }
        .ignoresSafeArea()
    }
}

private struct RegionLoupe: View {
    /// The same disk in the other pane, drawn fainter.
    var mirrored = false

    var body: some View {
        ZStack {
            Circle().fill(.white.opacity(0.06))
            Circle().stroke(.white.opacity(0.9), lineWidth: 1.5)
            Rectangle().fill(.white.opacity(0.8)).frame(width: 1, height: 9)
            Rectangle().fill(.white.opacity(0.8)).frame(width: 9, height: 1)
        }
        .frame(width: 2 * AppModel.regionRadius, height: 2 * AppModel.regionRadius)
        .opacity(mirrored ? 0.55 : 1)
        .shadow(color: .black.opacity(0.35), radius: 3)
    }
}

private struct DepthColumn: View {
    @Bindable var model: AppModel

    var body: some View {
        VStack(spacing: 9) {
            if let range = model.status.depthRange {
                ColorBarCard(range: range, focalPixels: model.status.focalPixels)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            if model.mode == .blend {
                BlendSlider(blend: $model.blend)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 5)
                    .glass(Capsule())
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
    }
}

private struct Header: View {
    @Bindable var model: AppModel

    private var preset: String {
        guard let preset = model.info?.preset else { return "S" }
        return preset.split(separator: "-").last.map(String.init)?.uppercased() ?? "S"
    }

    private var detail: String {
        guard let info = model.info, let resolution = model.currentResolution else { return "" }
        return "\(resolution.title) · W\(info.window)"
    }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            HStack(spacing: 9) {
                Image(systemName: "cube.transparent.fill")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [Turbo.color(0.95), Turbo.color(0.55)],
                            startPoint: .top, endPoint: .bottom))
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 5) {
                        Text("DEPTH PRO 2")
                            .font(.system(size: 11, weight: .bold, design: .rounded))
                            .tracking(1.4)
                        Text(preset)
                            .font(.system(size: 9, weight: .bold, design: .rounded))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(.white.opacity(0.16), in: Capsule())
                    }
                    Text(detail)
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.5))
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .glass(Capsule())

            Spacer(minLength: 0)

            VStack(alignment: .trailing, spacing: 1) {
                Text(model.status.processedFPS > 0
                    ? String(format: "%.1f FPS", model.status.processedFPS) : "— FPS")
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                Text(model.status.stepMs > 0
                    ? String(format: "%.0f ms", model.status.stepMs) : "on device")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
            }
            .padding(.horizontal, 11)
            .padding(.vertical, 9)
            .glass(Capsule())

            Button { model.showInfo = true } label: {
                Image(systemName: "info")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 38, height: 38)
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .glass(Circle())
            .accessibilityLabel("Model details")
        }
    }
}

private struct StatusPills: View {
    let model: AppModel

    var body: some View {
        VStack(spacing: 7) {
            if let error = model.status.error {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(Color.red.opacity(0.45), in: Capsule())
            }
            if let switching = model.status.switching {
                HStack(spacing: 8) {
                    ProgressView().tint(.white).scaleEffect(0.7)
                    Text(switching)
                }
                .font(.system(size: 11, weight: .medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .glass(Capsule())
            } else if model.status.calibrating {
                HStack(spacing: 8) {
                    ProgressView().tint(.white).scaleEffect(0.7)
                    Text("Calibrating depth range…")
                }
                .font(.system(size: 11, weight: .medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .glass(Capsule())
            }
            if model.mode == .compare, let region = model.status.region {
                HStack(spacing: 7) {
                    Image(systemName: "scope")
                    Text(String(format: "DP2 %.2f m · ARKit %.2f m · Δ %3.0f cm",
                                region.model, region.arkit, region.error * 100))
                        .monospacedDigit()
                    Button { model.clearRegion() } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.white.opacity(0.45))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Clear the selected area")
                }
                .font(.system(size: 11, weight: .medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .glass(Capsule())
            } else if model.mode == .compare, let error = model.status.arkitError {
                HStack(spacing: 6) {
                    Image(systemName: "ruler")
                    Text(String(format: "median |Δ| %.0f cm vs ARKit", error * 100))
                }
                .font(.system(size: 11, weight: .medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .glass(Capsule())
            }
        }
    }
}

private struct ColorBarCard: View {
    let range: ClosedRange<Double>
    let focalPixels: Double?

    private var middle: Double { (range.lowerBound * range.upperBound).squareRoot() }

    private func meters(_ value: Double) -> String {
        if value < 2 { return String(format: "%.0f cm", value * 100) }
        return value < 10 ? String(format: "%.1f m", value) : String(format: "%.0f m", value)
    }

    var body: some View {
        VStack(spacing: 7) {
            HStack {
                Text("DEPTH RANGE")
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .tracking(1.2)
                    .foregroundStyle(.white.opacity(0.55))
                Spacer()
                if let focalPixels {
                    Text("f \(Int(focalPixels)) px")
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.45))
                }
            }
            RoundedRectangle(cornerRadius: 3)
                .fill(LinearGradient(stops: Turbo.stops, startPoint: .trailing, endPoint: .leading))
                .frame(height: 6)
                .shadow(color: .black.opacity(0.35), radius: 3, y: 1)
            HStack {
                Text(meters(range.lowerBound))
                Spacer()
                Text(meters(middle))
                Spacer()
                Text(meters(range.upperBound))
            }
            .font(.system(size: 9, weight: .medium, design: .monospaced))
            .foregroundStyle(.white.opacity(0.7))
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .glass(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}

private struct BlendSlider: View {
    @Binding var blend: Double

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "camera")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white.opacity(0.6))
            Slider(value: $blend, in: 0.02...0.98)
                .tint(.white)
            Image(systemName: "square.stack.3d.up.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.white.opacity(0.6))
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Camera to depth blend")
    }
}

private struct Controls: View {
    @Bindable var model: AppModel

    var body: some View {
        HStack(spacing: 10) {
            HStack(spacing: 4) {
                ForEach(model.availableModes) { mode in
                    Button {
                        withAnimation(.snappy(duration: 0.24)) { model.setMode(mode) }
                    } label: {
                        Image(systemName: mode.icon)
                            .font(.system(size: 16, weight: .semibold))
                            .frame(width: 46, height: 42)
                            .background(model.mode == mode ? Color.white : Color.clear, in: Capsule())
                            .foregroundStyle(model.mode == mode ? Color.black : Color.white.opacity(0.72))
                            .contentShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(mode.label)
                }
            }
            .padding(5)
            .glass(Capsule())

            Spacer(minLength: 0)

            CircleButton(systemName: "scope") { model.recalibrate() }
                .accessibilityLabel("Recalibrate depth colors")
            CircleButton(systemName: "arrow.triangle.2.circlepath.camera") { model.flipCamera() }
                .disabled(model.mode == .compare)
                .opacity(model.mode == .compare ? 0.4 : 1)
                .accessibilityLabel("Switch camera")
        }
    }
}

private struct LaunchView: View {
    let message: String

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            RadialGradient(
                colors: [Turbo.color(0.8).opacity(0.28), .clear],
                center: .center, startRadius: 0, endRadius: 330)
                .ignoresSafeArea()
            VStack(spacing: 16) {
                DepthMark(size: 92)
                VStack(spacing: 5) {
                    Text("Depth Pro 2")
                        .font(.system(size: 30, weight: .bold, design: .rounded))
                    Text("Metric depth, on device")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(.white.opacity(0.55))
                }
                HStack(spacing: 9) {
                    ProgressView().tint(.white)
                    Text(message)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.white.opacity(0.7))
                }
                .padding(.top, 14)
            }
        }
    }
}

private struct FailureView: View {
    let message: String
    let retry: () -> Void
    let openSettings: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 18) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 40, weight: .semibold))
                    .foregroundStyle(Turbo.color(0.85))
                Text("Couldn’t start")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                Text(message)
                    .font(.system(size: 13))
                    .foregroundStyle(.white.opacity(0.6))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
                VStack(spacing: 10) {
                    Button("Open Settings", action: openSettings)
                        .buttonStyle(PillButtonStyle(prominent: true))
                    Button("Try Again", action: retry)
                        .buttonStyle(PillButtonStyle(prominent: false))
                }
                .padding(.top, 6)
            }
        }
    }
}

private struct PillButtonStyle: ButtonStyle {
    let prominent: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 14, weight: .semibold))
            .frame(width: 190, height: 44)
            .background(prominent ? Color.white : Color.white.opacity(0.12), in: Capsule())
            .foregroundStyle(prominent ? Color.black : Color.white)
            .opacity(configuration.isPressed ? 0.7 : 1)
    }
}

private struct InfoSheet: View {
    @Bindable var model: AppModel
    @Environment(\.dismiss) private var dismiss

    private var preset: String {
        guard let preset = model.info?.preset else { return "Depth Pro 2 S" }
        let size = preset.split(separator: "-").last.map(String.init)?.uppercased() ?? "S"
        return "Depth Pro 2 · \(size)"
    }

    private func note(for resolution: Resolution) -> String {
        let area = resolution.width * resolution.height
        if area < 854 * 480 { return "Fastest, least detail" }
        if area == 854 * 480 { return "Balanced" }
        return "Sharper edges, more compute"
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    HStack(spacing: 14) {
                        DepthMark(size: 64)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(preset)
                                .font(.system(size: 20, weight: .bold, design: .rounded))
                            Text("Every pixel’s distance, in meters, estimated entirely on this iPhone.")
                                .font(.system(size: 12))
                                .foregroundStyle(.secondary)
                        }
                    }

                    if model.models.count > 1 {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("MODEL")
                                .font(.system(size: 10, weight: .bold, design: .rounded))
                                .tracking(1.2)
                                .foregroundStyle(.secondary)
                            ForEach(model.models) { package in
                                choiceRow(
                                    package.label, package.note,
                                    font: .system(size: 15, weight: .semibold),
                                    selected: model.selectedModel == package
                                ) { model.selectModel(package) }
                            }
                        }
                        .disabled(model.status.switching != nil)
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("RESOLUTION")
                            .font(.system(size: 10, weight: .bold, design: .rounded))
                            .tracking(1.2)
                            .foregroundStyle(.secondary)
                        ForEach(model.resolutions) { resolution in
                            choiceRow(
                                resolution.title, note(for: resolution),
                                font: .system(size: 15, weight: .semibold, design: .monospaced),
                                selected: model.currentResolution == resolution
                            ) { model.select(resolution) }
                        }
                    }
                    .disabled(model.status.switching != nil)

                    VStack(spacing: 0) {
                        row("Temporal window", "\(model.info?.window ?? 4) frames")
                        Divider().overlay(.white.opacity(0.08))
                        row("Compute", model.info?.computeLabel ?? "—")
                        Divider().overlay(.white.opacity(0.08))
                        row("Focal length", model.status.focalPixels.map { "\(Int($0)) px" } ?? "—")
                    }
                    .padding(.horizontal, 14)
                    .background(.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 14))

                    Text("Depth is measured along the camera’s viewing direction. Tap the target button to refit the color range to the scene. Model weights are released by Apple under the Apple Machine Learning Research Model License, for research use only.")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(20)
            }
            .background(Color.black)
            .navigationTitle("Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.large])
        .presentationDragIndicator(.visible)
        .preferredColorScheme(.dark)
    }

    private func choiceRow(
        _ title: String, _ note: String, font: Font, selected: Bool, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(font)
                    Text(note)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if selected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.white)
                }
            }
            .padding(14)
            .background(.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }

    private func row(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title).font(.system(size: 13))
            Spacer()
            Text(value)
                .font(.system(size: 13, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 11)
    }
}
