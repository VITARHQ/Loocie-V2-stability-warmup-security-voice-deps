import SwiftUI
import AppKit

@main
struct LoocieCoreAIApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var engine = EngineManager.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(engine)
        }

        Settings {
            SettingsView()
                .environmentObject(engine)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        EngineManager.shared.stopOwnedEngine()
    }
}
