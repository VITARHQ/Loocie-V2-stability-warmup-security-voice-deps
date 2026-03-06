import Foundation
import SwiftUI
import Combine
import Security

final class EngineManager: ObservableObject {
    private static let debugLogPath = "/Volumes/LoocieCoreAI/Logs/loociecoreai-launch.log"

    nonisolated private static func debugLog(_ message: String) {
        let line = "[\(Date())] \(message)\n"
        let url = URL(fileURLWithPath: debugLogPath)
        let data = Data(line.utf8)
        if FileManager.default.fileExists(atPath: debugLogPath) {
            if let handle = try? FileHandle(forWritingTo: url) {
                _ = try? handle.seekToEnd()
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
            if apiKey.isEmpty {
                Self.deleteAPIKeyFromKeychain()
            } else {
                Self.saveAPIKeyToKeychain(apiKey)
            }
        }
    }

    private var pollTask: Task<Void, Never>?
    private var engineProcess: Process?
    private var enginePID: Int32?
    private var appStartedEngine: Bool = false

    private static let baseURLDefaultsKey = "LoocieCoreAI.baseURL"
    private static let keychainService = "HIP.LoocieCoreAI"
    private static let keychainAccount = "engine_api_key"


    private static func saveAPIKeyToKeychain(_ value: String) {
        guard let data = value.data(using: .utf8) else { return }

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: keychainAccount
        ]

        SecItemDelete(query as CFDictionary)

        var addQuery = query
        addQuery[kSecValueData as String] = data
        let status = SecItemAdd(addQuery as CFDictionary, nil)
        debugLog("KEYCHAIN save status= \(status)")
    }

    private static func loadAPIKeyFromKeychain() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: keychainAccount,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        debugLog("KEYCHAIN load status= \(status)")

        guard status == errSecSuccess,
              let data = result as? Data,
              let value = String(data: data, encoding: .utf8) else {
            return nil
        }

        return value
    }

    private static func deleteAPIKeyFromKeychain() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: keychainAccount
        ]
        let status = SecItemDelete(query as CFDictionary)
        debugLog("KEYCHAIN delete status= \(status)")
    }

    private init() {
        let savedBaseURL = UserDefaults.standard.string(forKey: Self.baseURLDefaultsKey)
        self.baseURLString = savedBaseURL?.isEmpty == false ? savedBaseURL! : "http://127.0.0.1:8080"

        if let keychainKey = Self.loadAPIKeyFromKeychain(), !keychainKey.isEmpty {
            self.apiKey = keychainKey
        } else {
            self.apiKey = ""
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
        Self.debugLog("STOP REQUEST appStartedEngine= \(appStartedEngine) enginePID= \(enginePID.map(String.init) ?? "nil")")

        guard appStartedEngine else {
            Self.debugLog("STOP SKIPPED: appStartedEngine is false")
            return
        }

        if let process = engineProcess, process.isRunning {
            Self.debugLog("STOP terminating Process pid= \(process.processIdentifier)")
            process.terminate()
        } else if let pid = enginePID {
            Self.debugLog("STOP terminating via kill pid= \(pid)")
            kill(pid, SIGTERM)
        } else {
            Self.debugLog("STOP FAILED: no process or pid")
        }

        engineProcess = nil
        enginePID = nil
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

        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let text = String(data: data, encoding: .utf8) {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    Self.debugLog("ENGINE STDERR: \(trimmed)")
                }
            }
        }

        outPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty, let text = String(data: data, encoding: .utf8) {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.contains("GET /health") || trimmed.contains("GET /status") {
                    return
                }
                if !trimmed.isEmpty {
                    Self.debugLog("ENGINE STDOUT: \(trimmed)")
                }
            }
        }

        do {
            Self.debugLog("ENGINE LAUNCH executable= \(process.executableURL?.path ?? "nil")")
            Self.debugLog("ENGINE LAUNCH cwd= \(process.currentDirectoryURL?.path ?? "nil")")
            Self.debugLog("ENGINE LAUNCH args= \(process.arguments ?? [])")
            try process.run()
            engineProcess = process
            enginePID = process.processIdentifier
            appStartedEngine = true
            Self.debugLog("ENGINE STARTED pid= \(process.processIdentifier)")
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
