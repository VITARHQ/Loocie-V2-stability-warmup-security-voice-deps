import Foundation
import SwiftUI
import Combine

final class EngineManager: ObservableObject {
    private static let debugLogPath = "/Volumes/LoocieCoreAI/Logs/loociecoreai-launch.log"

    private static func debugLog(_ message: String) {
        let line = "[\(Date())] \(message)\n"
        let url = URL(fileURLWithPath: debugLogPath)
        let data = Data(line.utf8)
        if FileManager.default.fileExists(atPath: debugLogPath) {
            if let handle = try? FileHandle(forWritingTo: url) {
                try? handle.seekToEnd()
                try? handle.write(contentsOf: data)
                try? handle.close()
            }
        } else {
            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try? data.write(to: url)
        }
        print(message)
    }

    static let shared = EngineManager()

    @Published var engineOnline: Bool = false
    @Published var modelOnline: Bool = false
    @Published var lastError: String = ""

    @Published var baseURLString: String = "http://127.0.0.1:8080" {
        didSet {
            UserDefaults.standard.set(baseURLString, forKey: Self.baseURLDefaultsKey)
        }
    }

    @Published var apiKey: String = "" {
        didSet {
            UserDefaults.standard.set(apiKey, forKey: Self.apiKeyDefaultsKey)
        }
    }

    private var pollTask: Task<Void, Never>?
    private var engineProcess: Process?
    private var appStartedEngine: Bool = false

    private let launcherPath = "/Volumes/LoocieCoreAI/LoocieCoreAI_Core/LoocieAI_V2_Master/LoocieCoreAI/macapp/scripts/START_ENGINE_FOREGROUND.sh"

    private static let baseURLDefaultsKey = "LoocieCoreAI.baseURL"
    private static let apiKeyDefaultsKey = "LoocieCoreAI.apiKey"

    private static func loadAPIKeyFromEngineEnv() -> String? {
        let path = "/Volumes/LoocieCoreAI/LoocieCoreAI_Core/LoocieAI_V2_Master/LoocieCoreAI/engine/.env"
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        for line in text.split(whereSeparator: \ .isNewline) {
            if line.hasPrefix("LOOCIE_INTERNAL_KEY=") {
                return String(line.split(separator: "=", maxSplits: 1).last ?? "")
            }
        }
        return nil
    }

    private init() {
        let savedBaseURL = UserDefaults.standard.string(forKey: Self.baseURLDefaultsKey)
        self.baseURLString = savedBaseURL?.isEmpty == false ? savedBaseURL! : "http://127.0.0.1:8080"

        let savedAPIKey = UserDefaults.standard.string(forKey: Self.apiKeyDefaultsKey)
        if let savedAPIKey, !savedAPIKey.isEmpty {
            self.apiKey = savedAPIKey
        } else {
            self.apiKey = Self.loadAPIKeyFromEngineEnv() ?? ""
        }

        Task {
            await ensureEngineRunning()
            startPolling()
        }
    }

    var baseURL: URL? { URL(string: baseURLString) }

    func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while let self = self, !Task.isCancelled {
                await self.refreshStatus()
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    @MainActor
    func refreshStatus() async {
        print("LoocieCoreAI baseURLString=", baseURLString)

        guard let baseURL = baseURL else {
            engineOnline = false
            modelOnline = false
            lastError = "Invalid base URL"
            return
        }

        do {
            _ = try await APIClient.shared.getHealth(baseURL: baseURL)
            engineOnline = true
        } catch {
            engineOnline = false
            modelOnline = false
            lastError = "Health failed: \(error.localizedDescription)"
            return
        }

        guard !apiKey.isEmpty else {
            modelOnline = false
            lastError = "Set X-API-Key in Settings."
            return
        }

        do {
            let st = try await APIClient.shared.getStatus(baseURL: baseURL, apiKey: apiKey)
            modelOnline = (st.model == "online")
            lastError = st.last_error ?? ""
        } catch {
            modelOnline = false
            lastError = "Status failed: \(error.localizedDescription)"
        }
    }

    func restartEngineUserInitiated() {
        Task {
            stopOwnedEngine()
            try? await Task.sleep(nanoseconds: 500_000_000)
            await ensureEngineRunning()
        }
    }

    func stopOwnedEngine() {
        guard appStartedEngine, let process = engineProcess else { return }

        if process.isRunning {
            process.terminate()
        }

        engineProcess = nil
        appStartedEngine = false

        Task { @MainActor in
            self.engineOnline = false
            self.modelOnline = false
            self.lastError = "Engine stopped."
        }
    }

    private func ensureEngineRunning() async {
        if await isHealthOK() {
            await MainActor.run {
                self.lastError = ""
            }
            return
        }

        await MainActor.run {
            self.lastError = "Launching engine..."
        }

        await MainActor.run {
            self.lastError = "Launching engine..."
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/Volumes/LoocieCoreAI/BuildCache/_Python/loocie-v2-venv/bin/python")
        process.currentDirectoryURL = URL(fileURLWithPath: "/Volumes/LoocieCoreAI/LoocieCoreAI_Core/LoocieAI_V2_Master/LoocieCoreAI/engine")
        process.arguments = [
            "-m", "uvicorn", "app.main:app",
            "--host", "127.0.0.1",
            "--port", "8080"
        ]

        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        process.terminationHandler = { proc in
            Task { @MainActor in
                if !self.engineOnline {
                    self.lastError = "Engine process exited (code: \(proc.terminationStatus))"
                }
            }
            Self.debugLog("ENGINE TERMINATED code= \(proc.terminationStatus)")
        }

        process.terminationHandler = { proc in
            Task { @MainActor in
                if !self.engineOnline {
                    self.lastError = "Engine process exited (code: \(proc.terminationStatus))"
                }
            }
            Self.debugLog("ENGINE TERMINATED code= \(proc.terminationStatus)")
        }

        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let text = String(data: data, encoding: .utf8) {
                Self.debugLog("ENGINE STDERR: \(text.trimmingCharacters(in: .whitespacesAndNewlines))")
            }
        }

        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let text = String(data: data, encoding: .utf8) {
                Self.debugLog("ENGINE STDOUT: \(text.trimmingCharacters(in: .whitespacesAndNewlines))")
            }
        }

        do {
            Self.debugLog("ENGINE LAUNCH executable= \(process.executableURL?.path ?? \"nil\")")
            Self.debugLog("ENGINE LAUNCH cwd= \(process.currentDirectoryURL?.path ?? \"nil\")")
            Self.debugLog("ENGINE LAUNCH args= \(process.arguments ?? [])")
            try process.run()
            engineProcess = process
            appStartedEngine = true
        } catch {
            await MainActor.run {
                self.lastError = "Failed to start engine: \(error.localizedDescription)"
            }
            Self.debugLog("ENGINE LAUNCH FAILED: \(error.localizedDescription)")
            return
        }

        let ready = await waitForHealthReady(timeoutSeconds: 15)
        await MainActor.run {
            self.engineOnline = ready
            if !ready {
                self.lastError = "Engine failed to become ready in time."
            }
        }
    }

    private func isHealthOK() async -> Bool {
        guard let baseURL = baseURL else { return false }
        do {
            _ = try await APIClient.shared.getHealth(baseURL: baseURL)
            return true
        } catch {
            return false
        }
    }

    private func waitForHealthReady(timeoutSeconds: Int) async -> Bool {
        let deadline = Date().addingTimeInterval(TimeInterval(timeoutSeconds))
        while Date() < deadline {
            if await isHealthOK() {
                return true
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        return false
    }
}
