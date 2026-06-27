---
title: "How to Run an LLM On-Device with LiteRT-LM: A Complete iOS Integration Tutorial"
date: 2026-06-28 12:00:00 +0700
categories: [AI, SLM]
tags: [on-device, litert, gemma, ios, swift, mobile, tutorial]
toc: true
---

*Step-by-step: from a `.litertlm` model file to a working SwiftUI app generating text on a real iPhone — with every layer, command, and gotcha.*

---

## What you'll build

A SwiftUI app that loads a small language model (we use **gemma-3-270m**, ~270M params) and generates text **fully on-device** using Google's **LiteRT-LM** runtime — no network, no API keys. By the end you'll understand every layer between a Swift `Button` and the model's matmuls, and you'll be able to swap in any `.litertlm` model.

**Audience:** AI engineers / mobile engineers comfortable with the command line. You don't need prior C++/Metal experience.

---

## The big picture: 4 layers

LiteRT-LM ships a **C API**. Swift can't call C++ directly, so we stack a few thin layers:

```
┌────────────────────────────────────────────┐
│ SwiftUI View + ViewModel                    │  your app
├────────────────────────────────────────────┤
│ Bridging header → GemmaBridge (Obj-C++)     │  NSString ↔ std::string
├────────────────────────────────────────────┤
│ GemmaEngine (C++)  →  GemmaEngine.xcframework │ wraps the C API
├────────────────────────────────────────────┤
│ CLiteRTLM.xcframework (LiteRT-LM C runtime) │  + model.litertlm
└────────────────────────────────────────────┘
```

Key insight: the **`.litertlm` file is self-contained** — it bundles the weights, the tokenizer, the chat template, and metadata (context length, special tokens). The runtime reads all of that from the file, so the integration is model-agnostic.

---

## Prerequisites

```bash
# macOS with Xcode 15+ (we used Xcode 26)
xcode-select --install

# Tooling
brew install xcodegen cmake

# An iPhone (for GPU/real perf) or just the simulator
```

You'll also need an Apple Developer account (a free personal team works for on-device installs).

---

## Step 1 — Get the LiteRT-LM runtime

LiteRT-LM publishes prebuilt iOS binaries on GitHub Releases. Fetch the xcframework (here, version `v0.12.0`):

```bash
VERSION=v0.12.0
curl -fL -o ios.zip \
  "https://github.com/google-ai-edge/LiteRT-LM/releases/download/${VERSION}/CLiteRTLM.xcframework.zip"
unzip -q ios.zip -d Frameworks/
# Also grab the C header so we can compile against it:
curl -fL -o include/litert_lm/c/engine.h \
  "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/${VERSION}/c/engine.h"
```

You now have `Frameworks/CLiteRTLM.xcframework` — a single static C-API binary with `ios-arm64` and `ios-arm64-simulator` slices.

> **What's inside?** The C API (`engine.h`) exposes engine settings, sessions/conversations, streaming generation, tokenize/detokenize, and a benchmark info struct. That's all you need.

---

## Step 2 — Get a model (`.litertlm`)

Two options:

1. **Use a prebuilt `.litertlm`** (e.g. from Google AI Edge / Kaggle / Hugging Face).
2. **Convert one yourself** with the AI Edge Torch / LiteRT-LM export pipeline (Python). Conversion quantizes the model and packs weights + tokenizer + chat template + metadata into the single `.litertlm` container.

Drop the result in `Resources/model.litertlm`. A gemma-3-270m int8/int4 build is ~270 MB.

> **Verify it loads:** the file starts with the magic bytes `LITERTLM` and contains sections like `LlmMetadataProto`, `HF_Tokenizer_Zlib`, and `TFLiteModel`. The metadata holds the chat template and the context length (ours: 1024 tokens).

---

## Step 3 — The C++ engine wrapper

The C API is verbose (create settings → create engine → create session config → create conversation → send message → parse JSON). Wrap it once in a small C++ class.

**`include/gemma_engine.h`**

```cpp
#ifndef GEMMA_ENGINE_H_
#define GEMMA_ENGINE_H_
#include <functional>
#include <string>

struct LiteRtLmEngine;
struct LiteRtLmConversation;

namespace gemma {

struct GenerationConfig {
    int   max_new_tokens = 256;
    float temperature    = 0.8f;
    int   top_k          = 40;
    float top_p          = 0.95f;
};

class GemmaEngine {
 public:
    GemmaEngine();
    ~GemmaEngine();

    // backend: "cpu" or "gpu". cache_dir: a writable dir for the kernel cache.
    bool Init(const std::string& model_path,
              const std::string& backend = "cpu",
              const std::string& cache_dir = "");

    std::string Generate(const std::string& prompt, const GenerationConfig& = {});

    using TokenCallback = std::function<void(const std::string& chunk, bool done)>;
    bool GenerateStream(const std::string& prompt, const GenerationConfig&, TokenCallback);

    const std::string& last_error() const { return last_error_; }

 private:
    bool RecreateConversation(const GenerationConfig&);
    LiteRtLmEngine*       engine_       = nullptr;
    LiteRtLmConversation* conversation_ = nullptr;
    std::string           last_error_;
};

}  // namespace gemma
#endif
```

**`src/gemma_engine.cpp`** (the important parts)

```cpp
#include "gemma_engine.h"
#include "litert_lm/c/engine.h"

namespace gemma {

// LiteRT-LM speaks JSON messages. Build a user turn:
static std::string UserMsg(const std::string& text) {
    return "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"" +
           JsonEscape(text) + "\"}]}";
}

bool GemmaEngine::Init(const std::string& model_path,
                       const std::string& backend,
                       const std::string& cache_dir) {
    // 1) Settings
    LiteRtLmEngineSettings* settings =
        litert_lm_engine_settings_create(model_path.c_str(), backend.c_str(),
                                         nullptr, nullptr);
    if (!settings) { last_error_ = "settings_create failed"; return false; }

    if (!cache_dir.empty())
        litert_lm_engine_settings_set_cache_dir(settings, cache_dir.c_str());

    // GPU + Gemma needs F32 activations (fp16 overflows -> garbage). See gotchas.
    if (backend == "gpu")
        litert_lm_engine_settings_set_activation_data_type(settings, /*F32=*/0);

    // 2) Engine
    engine_ = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);
    if (!engine_) { last_error_ = "engine_create failed"; return false; }

    return RecreateConversation(GenerationConfig{});
}

bool GemmaEngine::RecreateConversation(const GenerationConfig& cfg) {
    // 3) Session config: max tokens + sampler params
    LiteRtLmSessionConfig* sc = litert_lm_session_config_create();
    litert_lm_session_config_set_max_output_tokens(sc, cfg.max_new_tokens);
    litert_lm_session_config_set_apply_prompt_template(sc, true);  // use model's chat template
    LiteRtLmSamplerParams sp{};
    sp.type        = (cfg.temperature <= 0.f) ? kLiteRtLmSamplerTypeGreedy
                                              : kLiteRtLmSamplerTypeTopP;
    sp.top_k = cfg.top_k; sp.top_p = cfg.top_p; sp.temperature = cfg.temperature;
    litert_lm_session_config_set_sampler_params(sc, &sp);

    // 4) Conversation
    LiteRtLmConversationConfig* cc = litert_lm_conversation_config_create();
    litert_lm_conversation_config_set_session_config(cc, sc);
    conversation_ = litert_lm_conversation_create(engine_, cc);
    litert_lm_conversation_config_delete(cc);
    litert_lm_session_config_delete(sc);
    return conversation_ != nullptr;
}

bool GemmaEngine::GenerateStream(const std::string& prompt,
                                 const GenerationConfig& cfg, TokenCallback on_token) {
    if (!engine_ || !RecreateConversation(cfg)) return false;
    std::string msg = UserMsg(prompt);
    // 5) Stream — callback fires per token on a background thread
    int rc = litert_lm_conversation_send_message_stream(
        conversation_, msg.c_str(), nullptr, nullptr,
        [](void* data, const char* chunk, bool is_final, const char* err) {
            auto* cb = static_cast<TokenCallback*>(data);
            (*cb)(ExtractText(chunk ? chunk : ""), is_final);   // pull text out of JSON
        }, &on_token);
    return rc == 0;
}

}  // namespace gemma
```

The flow in one line: **settings → engine → session config (sampler) → conversation → `send_message_stream`**. Tokenization and chat templating happen inside the runtime (`apply_prompt_template = true`).

---

## Step 4 — Build the wrapper into an xcframework

We need both device and simulator slices. Use CMake's Xcode generator, one arch at a time, then fuse:

```bash
# build one slice
cmake -S . -B build/ios-os -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build build/ios-os --config Release --target gemma_engine

# (repeat with -DCMAKE_OSX_SYSROOT=iphonesimulator for the sim slice)

# assemble
xcodebuild -create-xcframework \
  -library build/ios-os/Release-iphoneos/libgemma_engine.dylib  -headers include \
  -library build/ios-sim/Release-iphonesimulator/libgemma_engine.dylib -headers include \
  -output dist/GemmaEngine.xcframework
```

Your `CMakeLists.txt` links `libgemma_engine.dylib` against `CLiteRTLM.framework`. Copy the result into the app:

```bash
cp -R dist/GemmaEngine.xcframework sample-ios-app/Frameworks/
```

(The sample bundles this whole thing in `scripts/bootstrap.sh` + `scripts/build_ios.sh` — one command.)

---

## Step 5 — The Obj-C++ bridge

Swift talks to C++ through an **Objective-C++** (`.mm`) class. Marshal `NSString` ↔ `std::string` and blocks ↔ `std::function`.

**`GemmaBridge.h`**

```objc
#import <Foundation/Foundation.h>
NS_ASSUME_NONNULL_BEGIN
@interface GemmaBridge : NSObject
- (BOOL)loadWithModelPath:(NSString *)modelPath
                  backend:(NSString *)backend
                 cacheDir:(nullable NSString *)cacheDir
    NS_SWIFT_NAME(load(modelPath:backend:cacheDir:));

- (void)streamPrompt:(NSString *)prompt
           maxTokens:(int)maxTokens
         temperature:(float)temperature
                topK:(int)topK
                topP:(float)topP
             onChunk:(void (^)(NSString *chunk, BOOL done))onChunk
    NS_SWIFT_NAME(stream(prompt:maxTokens:temperature:topK:topP:onChunk:));

@property (nonatomic, readonly, copy) NSString *lastError;
@end
NS_ASSUME_NONNULL_END
```

**`GemmaBridge.mm`**

```objc
#import "GemmaBridge.h"
#include "gemma_engine.h"

@implementation GemmaBridge {
    std::unique_ptr<gemma::GemmaEngine> _eng;
}
- (instancetype)init {
    if ((self = [super init])) _eng = std::make_unique<gemma::GemmaEngine>();
    return self;
}
- (BOOL)loadWithModelPath:(NSString *)p backend:(NSString *)b cacheDir:(NSString *)c {
    std::string cd = c ? std::string(c.UTF8String) : "";
    return _eng->Init(p.UTF8String, b.UTF8String, cd) ? YES : NO;
}
- (void)streamPrompt:(NSString *)prompt maxTokens:(int)n temperature:(float)t
                topK:(int)k topP:(float)p onChunk:(void(^)(NSString*, BOOL))onChunk {
    gemma::GenerationConfig cfg{n, t, k, p};
    _eng->GenerateStream(prompt.UTF8String, cfg,
        [onChunk](const std::string& chunk, bool done) {
            onChunk([NSString stringWithUTF8String:chunk.c_str()] ?: @"", done);
        });
}
@end
```

**Bridging header** (`*-Bridging-Header.h`) — one line so Swift sees it:

```objc
#import "GemmaBridge.h"
```

> **Threading:** the stream callback fires on the engine's worker thread. Marshal to the main queue before touching UI.

---

## Step 6 — The Swift layer

A `@MainActor` ViewModel loads the model off the main thread, then streams tokens.

```swift
@MainActor
final class ChatViewModel: ObservableObject {
    @Published var output = ""
    @Published var isReady = false
    private var engine: GemmaBridge?

    func warmUp() async {
        guard let path = Bundle.main.path(forResource: "model", ofType: "litertlm")
        else { return }
        // Writable cache dir — the .app bundle is read-only (see gotchas).
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
            .first?.appendingPathComponent("llm").path
        try? FileManager.default.createDirectory(atPath: cache ?? "",
                                                 withIntermediateDirectories: true)
        // Load on a background task — the model is hundreds of MB.
        let bridge = await Task.detached {
            let b = GemmaBridge()
            _ = b.load(modelPath: path, backend: "cpu", cacheDir: cache)
            return b
        }.value
        engine = bridge
        isReady = true
    }

    func send(_ prompt: String) {
        guard let engine else { return }
        output = ""
        DispatchQueue.global(qos: .userInitiated).async {
            engine.stream(prompt: prompt, maxTokens: 256, temperature: 0.2,
                          topK: 40, topP: 0.95) { chunk, done in
                DispatchQueue.main.async { self.output += chunk }   // marshal to UI
            }
        }
    }
}
```

A minimal view:

```swift
struct ContentView: View {
    @StateObject var vm = ChatViewModel()
    @State var input = "Translate to French: Good morning!"
    var body: some View {
        VStack {
            TextField("Prompt", text: $input)
            Button("Generate") { vm.send(input) }.disabled(!vm.isReady)
            ScrollView { Text(vm.output) }
        }
        .padding()
        .task { await vm.warmUp() }
    }
}
```

---

## Step 7 — Project configuration (xcodegen)

Generating the `.xcodeproj` by hand is painful; use **xcodegen** with a `project.yml`. The non-obvious settings are commented:

```yaml
name: MyLLMApp
options:
  deploymentTarget: { iOS: "16.0" }
settings:
  base:
    CLANG_CXX_LANGUAGE_STANDARD: "c++17"   # GemmaBridge.mm includes C++ headers
    CLANG_CXX_LIBRARY: "libc++"
    ENABLE_DEBUG_DYLIB: NO                  # single-binary debug; split dylib is rejected on-device
targets:
  MyLLMApp:
    type: application
    platform: iOS
    sources:
      - path: MyLLMApp
        excludes: [Info.plist]
      - path: Resources          # NOT a folder reference — a group, so signing enumerates it
        buildPhase: resources
    settings:
      base:
        SWIFT_OBJC_BRIDGING_HEADER: MyLLMApp/MyLLMApp-Bridging-Header.h
        HEADER_SEARCH_PATHS:
          - $(SRCROOT)/Frameworks/GemmaEngine.xcframework/ios-arm64/Headers
        OTHER_LDFLAGS: [-lc++]
        LD_RUNPATH_SEARCH_PATHS:
          - "@executable_path/Frameworks"
          - "@loader_path/Frameworks"
    dependencies:
      - framework: Frameworks/GemmaEngine.xcframework
        embed: true                # dynamic libs must be embedded + signed
        sign: true
      - framework: Frameworks/CLiteRTLM.xcframework
        embed: true
        sign: true
```

Generate the project:

```bash
xcodegen generate          # re-run whenever you add a source/resource file
```

---

## Step 8 — Build, install, run

**Simulator:**

```bash
xcodebuild -project MyLLMApp.xcodeproj -scheme MyLLMApp \
  -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' build
```

**Real device** (needed for GPU and real perf). Find your device + team, then build/install/launch:

```bash
# device id
xcrun devicectl list devices

# build with automatic signing
xcodebuild -project MyLLMApp.xcodeproj -scheme MyLLMApp \
  -destination 'id=<DEVICE_UDID>' -allowProvisioningUpdates \
  CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM=<TEAM_ID> build

# install + launch (works over USB *or* Wi-Fi if paired)
APP=~/Library/Developer/Xcode/DerivedData/.../MyLLMApp.app
xcrun devicectl device install app --device <UDID> "$APP"
xcrun devicectl device process launch --device <UDID> com.you.MyLLMApp
```

To read the runtime's internal logs (super useful for debugging):

```bash
xcrun devicectl device process launch --console --device <UDID> com.you.MyLLMApp
```

---

## Gotchas & troubleshooting (learn from my bruises)

| Symptom | Cause | Fix |
|---|---|---|
| Cold start wastes 1–2 s every launch | Kernel cache written next to the model in the **read-only** bundle | Pass a writable `cacheDir` (`Caches/`) |
| "Missing bundle ID" on install | `Resources/` added as a **folder reference**; signing skips it | Add it as a **group** (`buildPhase: resources`) |
| Install rejected on iOS device | Xcode split debug dylib | `ENABLE_DEBUG_DYLIB: NO` |
| Frameworks not found at runtime | Not embedded / rpath wrong | `embed: true, sign: true` + `@executable_path/Frameworks` rpath |
| GPU output is all `<pad>` (Gemma) | **fp16 activation overflow** | `set_activation_data_type(F32)` for GPU |
| UI freezes on launch | Model loaded on main thread | Load in `Task.detached` |
| Added a file, build can't see it | xcodegen uses explicit file refs | Re-run `xcodegen generate` |

**Debugging tip:** capture device logs with `--console`. LiteRT-LM logs the loaded sections, the chosen backend, activation type, prefill/decode status, and per-token warnings — it's how I found the fp16 overflow.

---

## Swapping models

Because the tokenizer and chat template live *inside* the `.litertlm`, switching models is mostly a **file swap**: replace `Resources/model.litertlm`, rebuild, run. Watch for:

- **Size vs device RAM** (a 1.5B model can be ~1 GB+ — risky on older phones).
- **Format/runtime version** compatibility (the loader logs the file's version).
- **fp16 safety on GPU** (Gemma needs F32; Qwen-class models are usually fine).

---

## Where to go next

- **Streaming UI** — you already get per-token callbacks; render them live.
- **CPU vs GPU** — flip the `backend` string and benchmark. Spoiler: for short prompts + long outputs, CPU often wins. I measured exactly where the crossover is in the companion post → [**GPU Isn't Always the Answer: Benchmarking CPU vs GPU for SLMs on Mobile**]({% post_url 2026-06-27-gpu-vs-cpu-slm-mobile %}).
- **Conversation memory** — keep the conversation object alive across turns instead of recreating it.

That's the whole pipeline: `.litertlm` → C runtime → C++ wrapper → Obj-C++ bridge → Swift → an LLM running in your pocket, offline.

---

## Related reading

- 📊 [**GPU Isn't Always the Answer: Benchmarking CPU vs GPU for Small Language Models on Mobile**]({% post_url 2026-06-27-gpu-vs-cpu-slm-mobile %}) — now that you've got a model running, this post measures CPU vs GPU on the prefill/decode split and gives a concrete rule for which backend to ship.

---

*Reference implementation: the `sample-ios-app` in this repo (gemma-3-270m, LiteRT-LM v0.12.0). `scripts/bootstrap.sh` automates Steps 1–7.*
