import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var engine: EngineManager

    var body: some View {
        Form {
            Section("Engine Connection") {
                LabeledContent("Base URL") {
                    TextField("http://127.0.0.1:8080", text: $engine.baseURLString)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 280)
                }

                LabeledContent("X-API-Key") {
                    SecureField("Stored in Keychain", text: $engine.apiKey)
                        .multilineTextAlignment(.trailing)
                        .frame(width: 280)
                }

                Button("Refresh Status") {
                    Task { await engine.refreshStatus() }
                }
            }

            Section("Storage") {
                Text("Base URL is stored locally. API key is stored in Keychain.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding(16)
        .frame(width: 560)
    }
}
