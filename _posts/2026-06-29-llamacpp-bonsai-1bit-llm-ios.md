---
title: "How to Run a 1-Bit LLM On-Device with llama.cpp: Deploying Bonsai on iOS"
date: 2026-06-29 10:00:00 +0700
categories: [AI, SLM]
tags: [on-device, llama-cpp, bonsai, 1-bit, ios, swift, mobile, tutorial]
toc: true
---

*Step-by-step: from a 1-bit `.gguf` model to a working SwiftUI app generating text on a real iPhone — with the bridge code, the build, and the gotchas that actually cost me hours (a reasoning model that wouldn't stop thinking, a GPU that produced garbage, and a 6 tok/s mystery).*

---

## What you'll build

A SwiftUI app that loads **Bonsai-1.7B** — a **1-bit (Q1_0)** quantized LLM, ~231 MB on disk — and generates text **fully on-device** via **llama.cpp**, with a Metal (GPU) / CPU toggle. No network, no API keys. By the end you'll have a drop-in Obj-C++ bridge over `llama.cpp`, understand why 1-bit saves *memory but not compute*, and know exactly which backend to pick per device.

This is the companion to my [LiteRT-LM integration tutorial]({% post_url 2026-06-27-litert-lm-ios-integration %}) — same app shell, a second engine. If you've read that one, you'll feel at home.

**Audience:** AI / mobile engineers comfortable with the command line. No prior C++/Metal experience required.

---

## The big picture: 3 layers

llama.cpp ships a clean **C API** (`llama.h`). That's simpler than the LiteRT-LM path — we don't need a separate C++ engine library; the bridge calls `llama.h` directly:

```
┌──────────────────────────────────────────────┐
│ SwiftUI View + ViewModel                      │  your app
│   ↳ InferenceEngine protocol (engine-agnostic)│
├──────────────────────────────────────────────┤
│ Bridging header → LlamaBridge (Obj-C++)       │  NSString ↔ std::string,
│                                               │  chat template, decode loop
├──────────────────────────────────────────────┤
│ llama.xcframework (llama.cpp + ggml + Metal)  │  + model.gguf
└──────────────────────────────────────────────┘
```

Key insight: like `.litertlm`, the **`.gguf` file is self-contained** — weights, tokenizer, *and* the chat template all live inside it. The runtime reads them from the file, so the integration is model-agnostic… with one big asterisk for reasoning models (see Step 4).

---

## Prerequisites

```bash
# macOS with Xcode 15+ (I used Xcode 26)
xcode-select --install

# Tooling
brew install xcodegen cmake

# An iPhone for real perf (the GPU/CPU story only matters on hardware)
```

A free personal Apple Developer team is enough for on-device installs.

---

## Step 1 — Build the llama.cpp xcframework

llama.cpp has an official `build-xcframework.sh` that compiles for all Apple platforms with the **Metal library embedded** (`GGML_METAL_EMBED_LIBRARY=ON`, so there's no loose `.metallib` to ship). We only need the iOS slices, so we build then trim:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
./build-xcframework.sh          # builds macOS/iOS/tvOS/visionOS (~a few min)

# Trim to iOS device + simulator only (~18 MB vs ~750 MB full):
xcrun xcodebuild -create-xcframework \
  -framework build-apple/llama.xcframework/ios-arm64/llama.framework \
  -framework build-apple/llama.xcframework/ios-arm64_x86_64-simulator/llama.framework \
  -output /path/to/app/Frameworks/llama.xcframework
```

> **Mainline or fork?** Bonsai's model card tells you to clone the *PrismML fork* "for the Q1_0 kernels." That instruction is **stale** — Q1_0 was upstreamed (`GGML_TYPE_Q1_0 = 41`, PR #21273, merged). Mainline llama.cpp loads Bonsai out of the box; I verified the fork's ARM kernel is byte-identical. Use mainline for a maintained base. (I scripted this as `scripts/build_llama_xcframework.sh` in the repo.)

The framework binary is a lean ~5.5 MB arm64 dylib with Metal baked in, plus `llama.h` / `ggml*.h` headers.

---

## Step 2 — Get the model (Bonsai 1-bit GGUF)

Grab a Q1_0 GGUF — the public one or your own finetune:

```bash
# ~231 MB
curl -fL -o Resources/model.gguf \
  "https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf?download=true"
```

What this model actually is, from its metadata:

| Property | Value |
|---|---|
| Architecture | **qwen3** (GQA 16/8 heads, SwiGLU, RoPE+YaRN, RMSNorm) |
| Params | 1.72 B |
| Quant | **Q1_0** — 1 bit/weight, group-128, shared fp16 scale → ~1.125 bpw |
| On disk | ~231 MB (≈14× smaller than fp16) |
| Context | 32k train (we'll cap it on-device) |

> **The 1-bit packing:** each weight is a single bit (`0 → −scale`, `1 → +scale`), 128 weights sharing one fp16 scale. That's why a 1.7B model fits in 231 MB. Remember this — it saves *memory*, and we'll see later it does **not** save compute.

---

## Step 3 — The Obj-C++ bridge (`LlamaBridge`)

Swift can't call `llama.h` (C/C++) directly, so we wrap it in an Obj-C++ class. Header — note the selectors mirror the LiteRT-LM bridge, so the Swift layer can treat either engine identically:

```objc
// LlamaBridge.h
@interface LlamaBridge : NSObject
- (BOOL)loadWithModelPath:(NSString *)modelPath
                  backend:(NSString *)backend          // @"cpu" / @"gpu"
                 cacheDir:(nullable NSString *)cacheDir // unused (parity)
                benchmark:(BOOL)benchmark
    NS_SWIFT_NAME(load(modelPath:backend:cacheDir:benchmark:));
- (void)streamPrompt:(NSString *)prompt maxTokens:(int)maxTokens
         temperature:(float)t topK:(int)k topP:(float)p
             onChunk:(void (^)(NSString *chunk, BOOL done))onChunk
    NS_SWIFT_NAME(stream(prompt:maxTokens:temperature:topK:topP:onChunk:));
- (nullable NSDictionary<NSString *, NSNumber *> *)lastBenchmark;
@end
```

The `.mm` does the real work. **Load** the model and context:

```objc++
#import <llama/llama.h>
#import <llama/ggml-backend.h>

llama_backend_init();   // once, global

llama_model_params mp = llama_model_default_params();
mp.n_gpu_layers = useGpu ? 999 : 0;   // 999 = offload all layers to Metal
mp.use_mmap     = true;               // mmap weights — keeps phys_footprint low
_model = llama_model_load_from_file(path.UTF8String, mp);

llama_context_params cp = llama_context_default_params();
cp.n_ctx     = 1024;                  // cap KV cache (see Step 5)
cp.n_batch   = 512;
cp.n_threads = cp.n_threads_batch = std::min(8u, std::thread::hardware_concurrency());
cp.no_perf   = false;                 // enable prefill/decode timing
_ctx   = llama_init_from_model(_model, cp);
_vocab = llama_model_get_vocab(_model);
```

**Generate** — apply the model's chat template, tokenize, then the decode loop (prefill the prompt in one batch, then sample one token at a time):

```objc++
// 1. Format with the model's embedded chat template
const char *tmpl = llama_model_chat_template(_model, nullptr);
llama_chat_message msg{ "user", userText.c_str() };
std::vector<char> buf(userText.size()*2 + 512);
int n = llama_chat_apply_template(tmpl, &msg, 1, /*add_ass=*/true,
                                  buf.data(), (int)buf.size());
std::string formatted(buf.data(), n);

// 2. Tokenize
std::vector<llama_token> toks(/* sized via a first call returning -count */);
llama_tokenize(_vocab, formatted.c_str(), formatted.size(),
               toks.data(), toks.size(), /*add_special=*/true, /*parse_special=*/true);

// 3. Sampler chain: top_k → top_p → temp → dist
llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topK));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

// 4. Decode loop
llama_token cur = 0;
llama_batch batch = llama_batch_get_one(toks.data(), toks.size());  // prefill
char piece[256];
for (int i = 0; i < maxTokens; ++i) {
    if (llama_decode(_ctx, batch) != 0) break;
    cur = llama_sampler_sample(smpl, _ctx, -1);
    if (llama_vocab_is_eog(_vocab, cur)) break;            // end-of-generation
    int np = llama_token_to_piece(_vocab, cur, piece, sizeof(piece), 0, false);
    onChunk(std::string(piece, np), false);               // stream it out
    batch = llama_batch_get_one(&cur, 1);                  // next: single token
}
llama_sampler_free(smpl);
```

**Benchmark for free** — `llama_perf_context` gives the prefill/decode split, which maps straight onto a dictionary Swift can read:

```objc++
llama_perf_context_data d = llama_perf_context(_ctx);
// prefillTps = d.n_p_eval / (d.t_p_eval_ms/1000)
// decodeTps  = d.n_eval   / (d.t_eval_ms /1000)
```

That's the whole engine. ~200 lines. But two non-obvious things will bite you — Steps 4 and 5.

---

## Step 4 — Make it stop "thinking" (the `<think>` gotcha)

First run, my output looked like this:

```
<think>
Okay, the user wants me to translate... let me consider the tone...
</think>
Xin chào, hôm nay bạn thế nào?
```

Bonsai is built on **Qwen3 — a reasoning model** — so it emits `<think>…</think>` before the answer. Annoying for a translator.

The twist: Bonsai's template is *supposed* to suppress this. Its generation prompt hardcodes an **empty think block**:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}
{%- endif %}
```

But `llama_chat_apply_template` **doesn't run Jinja** — it pattern-matches your template string to a built-in (ChatML/qwen), which ends at `<|im_start|>assistant\n` and **drops the empty think block**. So the model reasons on its own. (`--chat-template-kwargs '{"enable_thinking":false}'` is a `llama-server` flag and [unreliable anyway](https://github.com/ggml-org/llama.cpp/issues/20182); we're on the C API.)

The fix is to re-add what the template intended — append the empty think block ourselves after the assistant header:

```objc++
// Detect a reasoning model at load: _isThinking = _chatTmpl contains "<think>"
if (_isThinking && formatted.find("</think>") == std::string::npos) {
    formatted += "<think>\n\n</think>\n\n";   // the model's own no-think format
}
```

Now it answers directly. This isn't a hack — it's literally what the model's official template does.

---

## Step 5 — Make it fast on CPU (don't let ggml outsmart you)

My first benchmark: **decode 6 tok/s, and a 14-second freeze on launch.** Two self-inflicted wounds, both about backend scheduling.

**Wound 1 — Metal in CPU mode.** Even with `n_gpu_layers = 0`, ggml registers the Metal backend and the scheduler offloads ops to it, bouncing each layer CPU↔Metal. On an iPhone XS that produced **395 graph splits per decode token** (hundreds of sync stalls) *and* a one-time ~14 s Metal shader compile at launch.

**The fix — pin CPU mode to the CPU + BLAS devices and exclude Metal:**

```objc++
std::vector<ggml_backend_dev_t> devs;
if (!useGpu) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        auto t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU)
            continue;                  // skip Metal entirely
        devs.push_back(d);             // keep CPU + BLAS (Accelerate)
    }
    devs.push_back(nullptr);
    mp.devices = devs.data();
}
```

Result: decode graph splits **395 → 1**, the 14 s launch compile vanished, and **BLAS still accelerates prefill** — prefill jumped from **8 → 21 tok/s**.

**Wound 2 — thread count.** I assumed fewer threads (just the 2 performance cores) would help latency-bound decode. Wrong — measured: 2 threads = 4.7, 4 = 5.5, 6 = 6.3 tok/s. Decode is **compute-bound** and scales with cores, so use them all.

**Memory:** `n_ctx` drives the KV cache. At 32k it's ~3.7 GB (won't fit a 4 GB phone); cap it. At `n_ctx = 1024` the KV cache is ~112 MiB; weights are mmap'd. Peak RAM ≈ 0.5 GB.

---

## Step 6 — The Swift layer (one app, two engines)

Define a protocol both bridges conform to, and auto-pick the engine by which model file is bundled:

```swift
protocol InferenceEngine: AnyObject {
    func load(modelPath: String, backend: String, cacheDir: String?, benchmark: Bool) -> Bool
    func stream(prompt: String, maxTokens: Int32, temperature: Float, topK: Int32,
                topP: Float, onChunk: @escaping (String, Bool) -> Void)
    func lastBenchmark() -> [String: NSNumber]?
    var lastError: String { get }
}
extension GemmaBridge: InferenceEngine {}   // LiteRT-LM (.litertlm)
extension LlamaBridge: InferenceEngine {}   // llama.cpp (.gguf)

enum ModelRuntime {
    static func resolve() -> (runtime: ModelRuntime, path: String)? {
        if let p = ResourceLookup.path("model", ext: "gguf")     { return (.llamaCpp, p) }
        if let p = ResourceLookup.path("model", ext: "litertlm") { return (.liteRT, p) }
        return nil
    }
}
```

The view model holds `any InferenceEngine` and never knows which runtime is underneath.

---

## Step 7 — Project configuration (xcodegen)

Add the framework to `project.yml` (embed + sign — it's a dynamic framework):

```yaml
dependencies:
  - framework: Frameworks/llama.xcframework
    embed: true
    sign: true
settings:
  base:
    CLANG_CXX_LANGUAGE_STANDARD: "c++17"   # the .mm includes llama.h
    LD_RUNPATH_SEARCH_PATHS:
      - "@executable_path/Frameworks"
```

`LlamaBridge.mm` imports `<llama/llama.h>`, resolved from the embedded framework. Then `xcodegen generate`.

---

## Step 8 — Build, install, run

```bash
xcodebuild -project Gemma3Translator.xcodeproj -scheme Gemma3Translator \
  -configuration Debug -destination 'id=<DEVICE_UDID>' \
  -allowProvisioningUpdates CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM=<TEAM> build

xcrun devicectl device install app --device <UDID> <path>/Gemma3Translator.app
xcrun devicectl device process launch --console --device <UDID> com.example.gemma.Gemma3Translator
```

The `--console` capture is gold — that's where I read the load log:

```
general.architecture = qwen3
load_tensors: offloaded 0/29 layers to GPU
using device BLAS (Accelerate)
llama_kv_cache: CPU KV buffer size = 112.00 MiB
sched_reserve: graph splits = 282 (bs=512), 1 (bs=1)   ← clean decode
```

---

## Performance: the chip matters more than anything

Here's the part worth the whole post. Same model, same code, three chips:

| Device | Chip | CPU decode | GPU decode |
|---|---|---:|---:|
| iPhone XS | A12 (2018) | ~6 tok/s | ❌ garbage |
| iPhone 14 Pro | A16 (2022) | **17 tok/s** | **63 tok/s** |
| M4 Mac | M4 (2024) | 55 tok/s | 230 tok/s |

Three lessons fell out of this:

**1. The slowness is the *chip*, not the kernel — and I proved it.** I forced an M4 build to use the same scalar dot-product emulation the A12 uses: it still ran ~9× faster. The A12 is slow because it's a 2018 chip (~2.5 vs 4.4 GHz, 2 vs 4+ performance cores, narrower cores). The "missing `SDOT` instruction" everyone blames? I measured its impact at only **~18%** — a footnote, not the cause.

**2. 1-bit saves memory, not compute.** Decode on the A12 used only ~4% of memory bandwidth — it's compute-bound. The 1-bit weights expand to ±1 **int8** and run through the same int8 dot-product as any quantized model. The bit-packing is why it *fits*; it does nothing for *speed*.

**3. GPU is a per-device decision — and on old GPUs it's not even correct.** On the iPhone XS, the model fully offloaded to the A12 GPU (29/29 layers) but produced **garbage**: the A12 GPU lacks `simdgroup matrix mul` / `reduction`, which ggml-metal's kernels need. The *same* Metal kernels run correctly and ~3.7× faster than CPU on the A16. So:

> **Rule of thumb:** A12 (iPhone XS) → **CPU only**. A16+ (iPhone 14 Pro and newer) → **GPU**, it's ~4× faster *and* correct. Probe `hw.optional.arm.FEAT_DotProd` / the Metal `simdgroup` flags at runtime to choose automatically.

---

## Gotchas & troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Output starts with `<think>…</think>` | Reasoning model; Jinja no-think block dropped by `llama_chat_apply_template` | Append `<think>\n\n</think>\n\n` after the assistant header (Step 4) |
| Decode ~6 tok/s + 14 s launch freeze | Metal offload in CPU mode → 395 graph splits + shader compile | Pin `mp.devices` to CPU + BLAS, exclude GPU (Step 5) |
| GPU output is garbage on older device | A12 GPU lacks `simdgroup` matmul/reduction | Use CPU on A12; gate GPU on device features |
| Won't load: "unknown model architecture / type Q1_0" | Old llama.cpp without Q1_0 | Use mainline ≥ PR #21273 (or the prism fork) |
| App killed on launch (OOM) | KV cache too big at full context | Cap `cp.n_ctx` (1024 → ~112 MiB KV) |
| `dyld: Library not loaded: @rpath/llama.framework/llama` | Framework not embedded | Embed & Sign `llama.xcframework`; add `@executable_path/Frameworks` rpath |

---

## Swapping models

Any `.gguf` llama.cpp supports works — drop it in as `Resources/model.gguf` and rebuild. The bridge reads the architecture, chat template, and tokenizer from the file. For non-reasoning models, the `<think>` detection simply no-ops. For larger models, watch the KV cache and consider a smaller quant.

Want interactive speed on an *old* phone? The lever isn't the kernel — it's a **smaller model** (e.g. a ~0.6 B). The compute is what's expensive, and there's less of it.

---

## Where to go next

- **Auto-select the backend** per device (CPU on A12, GPU on A16+) using the feature probes above — so users never touch a toggle.
- **Stream to the UI** token-by-token (the bridge already calls `onChunk` per token).
- **Quantize your own finetune** to Q1_0 with the PrismML tooling and drop it in.

---

## Related reading

- 🛠️ [How to Run an LLM On-Device with LiteRT-LM: A Complete iOS Integration Tutorial]({% post_url 2026-06-27-litert-lm-ios-integration %}) — the sister post; same app shell, Google's runtime, `.litertlm` models.
- 📊 [GPU Isn't Always the Answer: CPU vs GPU for SLMs on Mobile]({% post_url 2026-06-27-gpu-vs-cpu-slm-mobile %}) — why the prefill/decode split and the *prompt-to-output ratio* decide your backend (and why "use the GPU" isn't a safe default).

---

*Reproducibility: Bonsai-1.7B Q1_0, mainline llama.cpp (Metal embedded), tested on iPhone XS (A12), iPhone 14 Pro (A16), and M4. Throughput from llama.cpp's `llama_perf_context`. Source: the SLM iOS sample app.*
</content>
</invoke>
