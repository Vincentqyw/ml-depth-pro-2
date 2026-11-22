# Depth Pro 2 for iPhone

An iPhone app that runs Apple's official Core ML Depth Pro 2 packages live in portrait or
landscape, with a camera/depth compare view, a metric depth color bar and a selectable temporal
window.

## Models

The official packages are not checked in (the repository ignores `*.mlpackage`). Download one or
both into `Models/`:

```bash
hf download apple/DepthPro2-coreml --include 'depthpro2-s_w4.mlpackage/*' --local-dir Models
hf download apple/DepthPro2-coreml --include 'depthpro2-b_w4.mlpackage/*' --local-dir Models
```

The app lists every `depthpro2-*.mlpackage` in `Models/`, fastest first, and switches between them
from the info sheet. S is the default on a fresh install; B is sharper but much slower (about 2–3
fps instead of 5–7 fps at `res_640x360`).

A package exported for one size (`depthpro2-<preset>_<H>x<W>_w<window>.mlpackage`) runs on the
Neural Engine: S with `-w 1` at `640x360` reaches about 25 fps on an iPhone 15 Pro Max, but has no
temporal window and cannot switch resolution. Multifunction packages (a function per resolution)
run on the GPU, where the stateful 4-frame window is affordable. The Neural Engine rejects
stateful single-function packages (`-w 4` fails to build an execution plan, error -14), so a
fast export is window 1:

```bash
uv run --extra coreml python -m demo.export stream -m depthpro2-s -w 1 -s 360x640 -o Models
```

The packages are Core ML spec v9 ML Programs: they need iOS 26. The app deploys to iOS 18, but the
models only load on iOS 26 or later.

## Runtime

The app starts each package at its default function; the info sheet switches the package, the
resolution (for multifunction packages) and the temporal window. The color range follows the scene
(5–95% depth quantiles, smoothed over time), so it adapts as you turn; the target button resets it.

The compute units follow the package: single-function packages load on the Neural Engine
(`.cpuAndNeuralEngine`), multifunction packages on the GPU (`.cpuAndGPU`). On iOS 26 the runtime
rejects multifunction packages on the Neural Engine (and `.all` picks it and fails the same way).
Switching to a package with larger functions restarts the camera at a format that covers them.

The model functions are landscape, so in portrait the camera frame is inferred sideways and the
displayable camera/depth result is rotated to the interface; the metric scale is unaffected.

## Build and run

```bash
open DepthPro2.xcodeproj
```

Choose the `DepthPro2` scheme and your iPhone. After editing `project.yml`, regenerate the project
with `xcodegen generate`. The app supports portrait and both landscape
orientations; the model functions themselves are landscape (`854×480` … `1280×720`).

With Xcode 16 and an iOS 26 device, build for a generic iOS device and install with
`devicectl` after enabling Developer Mode on the phone:

```bash
xcodebuild -scheme DepthPro2 -destination 'generic/platform=iOS' -allowProvisioningUpdates build
xcrun devicectl device install app --device <UDID> build/Build/Products/Debug-iphoneos/DepthPro2.app
xcrun devicectl device process launch --device <UDID> com.anureka.depthpro2
```
