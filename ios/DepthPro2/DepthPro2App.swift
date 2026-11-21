import SwiftUI
import UIKit

@main
struct DepthPro2App: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
                .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        }
    }
}
