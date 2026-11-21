import SwiftUI

enum Turbo {
    static func color(_ input: Float) -> Color {
        let x = min(max(input, 0), 1)
        let v4 = SIMD4<Float>(1, x, x * x, x * x * x)
        let v2 = SIMD2<Float>(v4.z * v4.z, v4.w * v4.z)
        let r = dot(v4, SIMD4(0.13572138, 4.61539260, -42.66032258, 132.13108234))
            + dot(v2, SIMD2(-152.94239396, 59.28637943))
        let g = dot(v4, SIMD4(0.09140261, 2.19418839, 4.84296658, -14.18503333))
            + dot(v2, SIMD2(4.27729857, 2.82956604))
        let b = dot(v4, SIMD4(0.10667330, 12.64194608, -60.58204836, 110.36276771))
            + dot(v2, SIMD2(-89.90310912, 27.34824973))
        return Color(red: unit(r), green: unit(g), blue: unit(b))
    }

    private static func dot(_ a: SIMD4<Float>, _ b: SIMD4<Float>) -> Float {
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    }

    private static func dot(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        a.x * b.x + a.y * b.y
    }

    static let stops = (0...24).map {
        Gradient.Stop(color: color(Float($0) / 24), location: Double($0) / 24)
    }

    private static func unit(_ value: Float) -> Double { Double(min(max(value, 0), 1)) }
}

extension View {
    func glass<S: InsettableShape>(_ shape: S) -> some View {
        background(.ultraThinMaterial, in: shape)
            .overlay(shape.strokeBorder(.white.opacity(0.14), lineWidth: 0.5))
    }
}

struct CircleButton: View {
    let systemName: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 18, weight: .semibold))
                .frame(width: 48, height: 48)
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .glass(Circle())
    }
}

struct DepthMark: View {
    var size: CGFloat = 96

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.26, style: .continuous)
                .fill(
                    LinearGradient(
                        stops: Array(Turbo.stops.reversed()),
                        startPoint: .topLeading, endPoint: .bottomTrailing))
            RoundedRectangle(cornerRadius: size * 0.26, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [.black.opacity(0.15), .black.opacity(0.55)],
                        startPoint: .top, endPoint: .bottom))
            Image(systemName: "square.stack.3d.up.fill")
                .font(.system(size: size * 0.44, weight: .semibold))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.35), radius: 8, y: 4)
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: size * 0.26, style: .continuous))
        .shadow(color: Turbo.color(0.9).opacity(0.35), radius: size * 0.25, y: size * 0.08)
    }
}
