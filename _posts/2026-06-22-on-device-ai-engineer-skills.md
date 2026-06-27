---
title: "Skills You Need to Become an Exceptional On-Device AI Engineer"
date: 2026-06-22 10:00:00 +0700
categories: [AI, SLM]
tags: [on-device, llm, quantization, llama-cpp, mobile]
toc: true
---

> After three months of hands-on research into on-device AI, I've come to realize this field is not simply about knowing how to code models — it sits at the intersection of machine learning, systems engineering, and hardware awareness. This post distills the skills that genuinely matter to do this job well.

---

## 1. Choosing the Right Inference Runtime for Your Deploy Environment

This is the first decision you make, and it shapes everything downstream.

**Android** offers more choices but also more complexity:
- **TFLite** — battle-tested, broad device support, the safe default for mid-range production
- **ONNX Runtime Mobile** — more flexible, works well when the model was trained in PyTorch
- **MediaPipe** — ideal for vision/audio tasks Google has already optimized
- **ExecuTorch** — Meta/PyTorch's new direction, moving fast
- **QNN SDK (Qualcomm)** — when you need to push directly to the Snapdragon NPU without going through an abstraction layer

**iOS** is simpler because Apple controls the full stack:
- **CoreML** — the default and best choice; directly backed by the Apple Neural Engine
- **MLX** — Apple Research's new framework, optimized for Apple Silicon, great for R&D
- **ONNX Runtime** — when cross-platform support between iOS and Android matters

**The rule of thumb:** prioritize the runtime that can delegate to the platform's native NPU or accelerator. CPU inference is a fallback, not the target.

---

## 2. Converting Models from Research Format to Inference Format

Data scientists train in PyTorch, JAX, or TensorFlow. The AI engineer's job is to convert those models into a format the inference runtime can actually run — **without losing accuracy**.

**Common pipelines:**

```
PyTorch  →  ONNX           →  TFLite / ORT / CoreML
JAX      →  SavedModel     →  TFLite
TF       →  SavedModel     →  TFLite
PyTorch  →  ExecuTorch        (via torch.export)
PyTorch  →  CoreML            (via coremltools)
PyTorch  →  GGUF              (via llama.cpp/convert_hf_to_gguf.py)
```

The last pipeline — HuggingFace checkpoint → GGUF — is the most common path when working with on-device LLMs. **llama.cpp** provides `convert_hf_to_gguf.py` for this step, and GGUF has become the de facto format for LLM inference on CPU and mobile GPU, thanks to its support for mmap loading (no need to load the entire model into RAM at once).

**Skills you need:**
- `torch.onnx.export`, `coremltools`, `tf2onnx`, `ai_edge_torch`
- Understanding **shape inference** and **dynamic shapes** — many tools require fixed input shapes
- Knowing when to use `torch.jit.trace` vs. `torch.export`, and when each one fails
- Checking numerical equivalence after conversion — the converted model's output must match the original within an acceptable tolerance

**Common mistakes:** tracing with the wrong input shape, exporting a model still in training mode instead of eval mode, not handling stateful models (RNNs, streaming ASR).

---

## 3. Evaluating Whether a Model Is Feasible to Deploy on Mobile

Before you start converting or optimizing, you need to answer one question: **can this model actually be deployed?**

**Key metrics to assess:**

| Metric | Practical threshold (mid-range device) |
|--------|----------------------------------------|
| Parameters | < 10M for real-time, < 100M for batch |
| Peak RAM | < 150MB (Android), < 200MB (iOS) |
| On-disk model size | < 20MB ideal, < 100MB acceptable |
| Inference latency | < 100ms for interactive, < 500ms for background |
| FLOPs | < 300M FLOPs/inference for real-time camera |

**Theory diverges significantly from reality.** Peak RAM during actual inference can be 2–3× the model size due to activation buffers. FLOPs don't reflect memory bandwidth bottlenecks — and for LLMs, memory bandwidth is often the real bottleneck, not compute.

One important edge case: if you're evaluating an LLM built on **BitNet 1.58** (ternary weights {-1, 0, +1}), the entire memory and compute math changes. Matrix multiplication becomes pure addition and subtraction, memory bandwidth drops 8–16× vs. FP16. The thresholds in the table above don't apply — you need to benchmark directly on real hardware.

**Thermal throttling** is consistently underestimated: a mid-range device can drop performance by 40–60% after 30 seconds of continuous inference due to heat. You need to test sustained performance, not just burst performance.

**Benchmark on the right device tier.** Always test on your "p80 device" — the device at the 80th percentile of your user base, not a Pixel 9 or iPhone 16. A model that runs beautifully on a flagship may be unusable on a Redmi Note 12.

---

## 4. Fine-tuning and Quantizing Models

These are the two primary tools for bridging the gap between research models and production models.

**Quantization — from theory to practice:**

Post-Training Quantization (PTQ) is the starting point: fast, no retraining required, typically 1–3% accuracy loss. Quantization-Aware Training (QAT) is more expensive but recovers accuracy better by inserting fake quantization nodes into the training graph. INT8 is the standard; INT4 is becoming increasingly common for on-device LLMs.

For LLMs, **GGUF quantization via llama.cpp** is the most practical PTQ approach available today. Rather than a single threshold, GGUF offers a spectrum of quantization levels to choose from based on your device's constraints:

| GGUF Level | Bits/weight | When to use |
|------------|-------------|-------------|
| Q2_K | ~2.6 bit | Severely RAM-constrained, acceptable quality loss |
| Q4_K_M | ~4.5 bit | Sweet spot — best balance of size and quality |
| Q5_K_M | ~5.5 bit | When RAM headroom exists and quality matters more |
| Q8_0 | ~8 bit | Near-lossless, use for debugging or benchmarking |

Knowing which level to choose for a given device tier is practical knowledge — it won't appear in a textbook.

**But quantization has limits.** All of the above approaches start from a trained FP32 model and reduce precision afterward. **BitNet b1.58** (Microsoft Research) asks a fundamentally different question: *why not train from scratch with ternary weights {-1, 0, +1}?*

The result: no multiplications in matrix operations — only additions and subtractions. Memory bandwidth drops 8–16×. Energy consumption drops significantly. This is not quantizing a pre-trained model; it's a completely different training paradigm, with its own inference runtime (`bitnet.cpp`). llama.cpp also supports BitNet inference via `-IQ1_S` format for benchmarking comparisons.

BitNet 1.58 is still early adoption as of mid-2026 — the model ecosystem is not yet as broad as GGUF. But if you're building a long-term on-device LLM pipeline, this is a direction worth tracking closely.

**Fine-tuning for on-device:**
- Fine-tune on representative domain data before quantizing — it makes models more robust to quantization noise
- **LoRA / QLoRA** for LLMs: drastically reduces the memory footprint during fine-tuning
- Always have a good calibration dataset for PTQ — data must represent real-world input distribution

---

## 5. Accelerator Targeting — The Skill That Separates Good from Great

This is where most engineers drop the ball, and where the biggest performance gains live.

Every platform has a dedicated AI accelerator:
- **iOS** → Apple Neural Engine (ANE) — extremely fast, but requires the right model format and ops
- **Android Qualcomm** → Hexagon DSP / Qualcomm AI Engine
- **Android MediaTek** → APU (AI Processing Unit)
- **Android Samsung Exynos** → NPU

**The problem:** not every layer can run on the NPU. When an op is unsupported, it falls back to CPU — and the data copy overhead between CPU and NPU (memory transfer) can sometimes be worse than running everything on CPU in the first place.

Same model: CPU → 150ms, GPU → 40ms, NPU → 8ms. That's why accelerator targeting matters this much.

**llama.cpp** is the clearest practical example of how a runtime handles multi-accelerator targeting: same codebase, build with `LLAMA_METAL=1` for Apple Silicon (ANE + GPU), `LLAMA_VULKAN=1` for Android GPU, `LLAMA_BLAS=1` for CPU-optimized path. This is a pattern worth studying — one model, multiple hardware targets, no need to rewrite inference logic.

**Skills you need:**
- Checking delegate compatibility before building (TFLite Delegate API, CoreML model validation)
- Understanding model partitioning: which parts run on the NPU, which parts fall back
- Reshaping model architectures to maximize NPU utilization (avoid dynamic shapes, avoid unsupported ops)

---

## 6. Operator Compatibility — Know Before You Commit

Not all PyTorch/TF operators are supported on every runtime. Discovering this after you've already optimized the model is expensive.

**Real examples:**
- TFLite doesn't support `tf.math.top_k` with dynamic k
- CoreML doesn't support certain custom attention variants
- Most runtimes don't support complex control flow (if/while inside the model graph)

**The right process:**
1. Check the target runtime's operator support list **upfront**
2. Use TFLite Model Analyzer, CoreML Validator, or ONNX checker to detect unsupported ops **before** the data science team finalizes the architecture
3. If possible, join architecture design reviews to prevent incompatible ops from being used in the first place

**Ideally:** the on-device AI engineer is in the room during model design, not just at the receiving end when the model is already done.

---

## 7. Device-Native Profiling — Measure, Don't Assume

FLOPs and parameter count are theoretical estimates. Real latency needs to be measured on real hardware.

**Toolchain by platform:**

**Android:**
- **Android GPU Inspector** — profile GPU/NPU execution, identify bottlenecks by layer
- **Snapdragon Profiler** — Qualcomm-specific, DSP utilization visibility
- **Android Studio Profiler** — RAM, CPU, battery during inference
- **systrace / perfetto** — low-level timing analysis

**iOS:**
- **Xcode Instruments → Core ML Instrument** — see how long each layer takes, and where it runs (CPU/GPU/ANE)
- **Metal Performance HUD** — GPU frame timing
- **Energy Log** — battery impact

**For on-device LLMs:** the tools above measure per-inference latency. With LLMs, the more meaningful metric is **tokens per second** — the throughput of the decode loop. `llama-bench` (built into llama.cpp) is the standard tool for measuring this directly on-device, with clear output broken down by quantization level and backend.

**What to measure:**
- **Latency** — time to first result, P50/P95 (not just average)
- **Sustained throughput** — performance after 30 seconds, 2 minutes of continuous inference
- **Memory watermark** — peak RAM throughout the full inference cycle
- **Battery drain** — how much battery does 10 minutes of the inference loop consume?
- **Thermal behavior** — does the device throttle? When?

---

## 8. Pruning and Knowledge Distillation — Beyond Quantization

Quantization reduces precision. But sometimes you need to reduce the **model architecture itself** to hit your latency target.

**Structured Pruning:**
- Remove entire channels, attention heads, or layers rather than individual weights
- Structured pruning → genuinely smaller model → faster inference on real hardware
- Unstructured pruning (sparse weights) looks good on paper but very few hardware implementations actually exploit sparsity effectively

**Knowledge Distillation:**
- Train a smaller student model to learn from the outputs of a larger teacher model
- Results are typically better than training a small model from scratch at the same size
- Especially effective when chained: distillation → pruning → quantization

**Typical optimization order:**
```
Teacher model (large)
    ↓ Knowledge Distillation
Student model (smaller architecture)
    ↓ Fine-tune on domain data
    ↓ Structured Pruning
    ↓ QAT (Quantization-Aware Training)
    ↓ Convert to inference format
    ↓ Deploy
```

---

## 9. On-Device LLMs — The Field That's Exploding

With Phi-3-mini, Gemma, and Llama-3.2 now in production, on-device LLMs are no longer experimental. This is a distinct skillset that on-device AI engineers need to develop — and it pulls in virtually every skill covered above, but with unique challenges.

**LLM-specific challenges:**
- **KV Cache management** — caching key/value attention states to avoid recomputation, but cache size grows linearly with context length and eats RAM fast
- **Prefill vs. decode latency** — prefill (processing the prompt) and decode (generating each token) have different bottlenecks and need to be optimized separately
- **Speculative decoding** — a small draft model predicts tokens, the large model verifies — 2–3× throughput gain; llama.cpp supports this natively
- **Context window trade-offs** — longer context = more RAM, higher latency; must match the use case
- **Streaming output** — much better UX to stream tokens one at a time; requires pipeline design to support it

When working with on-device LLMs, **llama.cpp** and **bitnet.cpp** are the two runtimes to understand first — not because they're the only options, but because they are the reference implementations that most other frameworks learn from or integrate. Understanding how llama.cpp handles quantization, backend delegation, and KV cache is the foundation for evaluating any other LLM runtime.

**Current ecosystem:**
- **llama.cpp** — primary runtime for GGUF, CPU + GPU + NPU, cross-platform
- **bitnet.cpp** — dedicated runtime for BitNet 1.58 models
- **ExecuTorch** — PyTorch's official mobile LLM runtime, deep Android/iOS native integration
- **MLC LLM** — compiles LLMs for specific hardware targets via TVM backend
- **MediaPipe LLM Inference API** — Google's high-level API, less control but easier to integrate

---

## 10. Native SDK Integration

A great model that can't be integrated into an app is worthless. On-device AI engineers need to write glue code.

**Android:**
- JNI / NDK — calling a C++ inference engine (like llama.cpp) from Kotlin/Java
- TFLite Java/Kotlin API, ONNX Runtime Android API
- CameraX + ML Kit integration for real-time camera pipelines

**iOS:**
- CoreML Swift API — loading models, running predictions, handling async inference
- Vision framework — wrapping CoreML models for image tasks
- AVFoundation — camera/audio pipeline for real-time inference

**Things to know:**
- Threading: inference should almost always run on a background thread, never blocking the main thread
- Memory management: especially critical on iOS under ARC
- Error handling: model load failures, inference timeouts, low-memory warnings

---

## 11. Model Update Pipeline — OTA Without an App Release

This is consistently overlooked, yet critical in production: **how do you update a model without shipping a new app version?**

**The problem:** iOS App Store review takes 1–7 days. If every model improvement requires an app release, your iteration cycle is painfully slow.

**The solution stack:**
- **Remote model serving** — host models on a CDN, download and cache on-device
- **Model versioning** — version models independently of the app version
- **Incremental updates** — download only the diff rather than the full model
- **Rollback mechanism** — when a new model has issues, fall back to the previous version
- **A/B testing** — ship the new model to 10% of users first, measure, then roll out fully

**Firebase ML, AWS ML Models, or a custom CDN** are the most common approaches for this pattern.

---

## 12. Testing and Validation Pipeline

After every optimization step (quantize, prune, convert), you need to validate that the model is still correct.

**Accuracy regression testing:**
- Compare outputs of the original model vs. the converted/quantized model on a test set
- Define acceptable thresholds: e.g., accuracy drop < 1%, cosine similarity > 0.99
- Automate this after every conversion step in your CI pipeline

**Latency benchmarking:**
- Benchmark across multiple device tiers, not just one device
- Run warm-up iterations before measuring (JIT compilation, cache warm-up)
- Measure P50, P95, P99 — not just the average

**Integration testing:**
- Test the entire pipeline: camera frame → preprocess → inference → postprocess → UI update
- Test edge cases: low-light input, unusual aspect ratios, out-of-distribution input

---

## Putting It Together: A Practical Roadmap

```
Months 1–2: Foundation
├── Runtime selection & setup (TFLite, CoreML, ONNX RT)
├── Model conversion pipeline (PyTorch/TF/HF → mobile format, GGUF)
└── Basic quantization (PTQ INT8, GGUF Q4_K_M)

Months 3–4: Optimization
├── Accelerator targeting (NPU delegate, llama.cpp backends)
├── Profiling with device-native tools + llama-bench
├── Operator compatibility check workflow
└── Knowledge distillation & structured pruning

Months 5–6: Production
├── Native SDK integration (JNI/NDK, CoreML Swift)
├── Model update pipeline (OTA)
├── Testing & validation automation
└── On-device LLMs: KV cache, speculative decoding, BitNet 1.58

Ongoing
└── Track hardware releases, new runtime versions, emerging architectures
```

**The skills with the highest impact that fewest engineers do well:** accelerator targeting and device-native profiling. Get these two right and you're already in the top 10% of the field.

---

*On-device AI is one of the rare disciplines where you need to understand ML, systems engineering, and hardware all at once to do the job well. That's exactly why great practitioners are rare — and exactly why they're compensated accordingly.*