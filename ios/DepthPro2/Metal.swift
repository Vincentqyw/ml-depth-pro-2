import CoreVideo
import Metal

struct GPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("This iPhone has no Metal device.")
        }
        self.device = device
        self.queue = queue
        library = try device.makeLibrary(source: Self.shaderSource, options: nil)
    }

    func privateTexture(_ format: MTLPixelFormat, width: Int, height: Int) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw EngineError("Cannot allocate a \(width)×\(height) texture.")
        }
        return texture
    }

    func texture(_ buffer: CVPixelBuffer, format: MTLPixelFormat, usage: MTLTextureUsage) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: CVPixelBufferGetWidth(buffer),
            height: CVPixelBufferGetHeight(buffer), mipmapped: false)
        descriptor.usage = usage
        guard let surface = CVPixelBufferGetIOSurface(buffer)?.takeUnretainedValue(),
              let texture = device.makeTexture(descriptor: descriptor, iosurface: surface, plane: 0)
        else { throw EngineError("The camera frame is not IOSurface-backed.") }
        return texture
    }

    private static let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct ResizeUniforms {
            float2 sourceSize;
            float2 scale;
            float2 translate;
        };

        kernel void resize_split(texture2d<half, access::sample> source [[texture(0)]],
                                 texture2d<half, access::write> camera [[texture(1)]],
                                 texture2d<half, access::write> planar [[texture(2)]],
                                 constant ResizeUniforms &u [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
            uint w = camera.get_width();
            uint h = camera.get_height();
            if (gid.x >= w || gid.y >= h) return;
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            float2 destination = float2(gid) + 0.5;
            float2 uv = ((destination - u.translate) / u.scale) / u.sourceSize;
            half4 c = clamp(source.sample(s, uv), 0.0h, 1.0h);
            camera.write(half4(c.rgb, 1.0h), gid);
            planar.write(half4(c.r), gid);
            planar.write(half4(c.g), uint2(gid.x, gid.y + h));
            planar.write(half4(c.b), uint2(gid.x, gid.y + 2 * h));
        }

        struct Varyings { float4 position [[position]]; float2 uv; };

        vertex Varyings fullscreen(uint vid [[vertex_id]]) {
            float2 p = float2((vid << 1) & 2, vid & 2);
            return { float4(p * 2.0 - 1.0, 0.0, 1.0), float2(p.x, 1.0 - p.y) };
        }

        // Turbo colormap, polynomial approximation by Ruofei Du of Anton Mikhailov's design:
        // https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
        // Copyright 2019 Google LLC. SPDX-License-Identifier: Apache-2.0
        float3 turbo(float x) {
            const float4 r4 = float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
            const float4 g4 = float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
            const float4 b4 = float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
            const float2 r2 = float2(-152.94239396, 59.28637943);
            const float2 g2 = float2(4.27729857, 2.82956604);
            const float2 b2 = float2(-89.90310912, 27.34824973);
            float4 v4 = float4(1.0, x, x * x, x * x * x);
            float2 v2 = v4.zw * v4.z;
            return saturate(float3(dot(v4, r4) + dot(v2, r2),
                                   dot(v4, g4) + dot(v2, g2),
                                   dot(v4, b4) + dot(v2, b2)));
        }

        struct Uniforms {
            float3 scaleNearFar;  // f_px / width, near meters, far meters
            float mode;           // 0 camera, 1 blend, 2 depth
            float blend;          // camera to depth mix in mode 1
            float2 uvScale;       // aspect-fill crop in display space
            int quarterTurns;     // clockwise display rotation of the landscape model
        };

        fragment half4 display(Varyings in [[stage_in]],
                               texture2d<half> camera [[texture(0)]],
                               texture2d<float> inverseDepth [[texture(1)]],
                               constant Uniforms &u [[buffer(0)]]) {
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            float2 uv = (in.uv - 0.5) * u.uvScale + 0.5;
            if (u.quarterTurns == 1) uv = float2(uv.y, 1.0 - uv.x);
            else if (u.quarterTurns == 2) uv = 1.0 - uv;
            else if (u.quarterTurns == 3) uv = float2(1.0 - uv.y, uv.x);
            half3 cam = camera.sample(s, uv).rgb;
            float meters = u.scaleNearFar.x / max(inverseDepth.sample(s, uv).r, 1e-6);
            float t = 1.0 - saturate(log(meters / u.scaleNearFar.y) / log(u.scaleNearFar.z / u.scaleNearFar.y));
            half3 depth = half3(turbo(t));
            half3 rgb = u.mode < 0.5 ? cam : (u.mode < 1.5 ? mix(cam, depth, half(u.blend)) : depth);
            return half4(rgb, 1.0h);
        }
        """
}

func makeHalfSurface(width: Int, height: Int) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let attributes: [CFString: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey: [String: Any](),
        kCVPixelBufferMetalCompatibilityKey: true,
    ]
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent16Half,
        attributes as CFDictionary, &buffer)
    guard status == kCVReturnSuccess, let buffer else {
        throw EngineError("Cannot allocate a \(width)×\(height) depth surface.")
    }
    return buffer
}

func fillScale(from source: (width: Int, height: Int), to target: (width: Int, height: Int)) -> Double {
    max(Double(target.width) / Double(source.width), Double(target.height) / Double(source.height))
}

extension GPU: @unchecked Sendable {}
