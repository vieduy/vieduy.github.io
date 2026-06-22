```markdown
---
title: Kỹ Năng Cần Có Để Trở Thành AI Engineer On-Device Xuất Sắc
date: 2026-06-22 10:00:00 +0700
categories: [AI, SLM]
---
# Kỹ Năng Cần Có Để Trở Thành AI Engineer On-Device Xuất Sắc

> Sau 3 tháng nghiên cứu thực chiến về on-device AI, tôi nhận ra rằng đây không phải là ngành chỉ cần biết code model — mà là giao điểm của ML, systems engineering, và hardware awareness. Bài viết này tổng hợp những kỹ năng thực sự cần thiết để làm tốt công việc này.

---

## 1. Chọn Inference Runtime Phù Hợp Với Deploy Environment

Đây là kỹ năng đầu tiên và cũng là quyết định ảnh hưởng đến toàn bộ pipeline sau đó.

**Android** có nhiều lựa chọn hơn nhưng cũng phức tạp hơn:
- **TFLite** — battle-tested, hỗ trợ rộng, phù hợp cho production với device tầm trung
- **ONNX Runtime Mobile** — linh hoạt hơn, tốt khi model được train bằng PyTorch
- **MediaPipe** — phù hợp cho các task vision/audio đã được Google optimize sẵn
- **ExecuTorch** — hướng đi mới từ Meta/PyTorch, đang phát triển nhanh
- **QNN SDK (Qualcomm)** — khi cần push thẳng xuống Snapdragon NPU, không qua abstraction layer

**iOS** đơn giản hơn nhờ Apple kiểm soát hardware:
- **CoreML** — lựa chọn mặc định và tối ưu nhất, được Apple Neural Engine hỗ trợ trực tiếp
- **MLX** — framework mới từ Apple Research, tối ưu cho Apple Silicon, phù hợp R&D
- **ONNX Runtime** — khi cần cross-platform giữa iOS và Android

**Nguyên tắc chọn runtime:** ưu tiên runtime nào có thể delegate xuống NPU/accelerator native của platform đó. CPU inference là fallback, không phải mục tiêu.

---

## 2. Convert Model Từ Research Format Sang Inference Format

Data scientist train model bằng PyTorch, JAX, hoặc TensorFlow. Nhiệm vụ của AI Engineer là chuyển đổi sang format mà inference runtime có thể chạy được — và làm điều này **mà không mất accuracy**.

**Pipeline phổ biến:**

```
PyTorch  →  ONNX           →  TFLite / ORT / CoreML
JAX      →  SavedModel     →  TFLite
TF       →  SavedModel     →  TFLite
PyTorch  →  ExecuTorch        (via torch.export)
PyTorch  →  CoreML            (via coremltools)
PyTorch  →  GGUF              (via llama.cpp/convert_hf_to_gguf.py)
```

Pipeline cuối cùng — HuggingFace checkpoint → GGUF — là con đường phổ biến nhất khi làm việc với LLM on-device. **llama.cpp** cung cấp script `convert_hf_to_gguf.py` để thực hiện bước này, và GGUF trở thành format de facto cho LLM inference trên CPU và mobile GPU nhờ hỗ trợ mmap loading (không cần load toàn bộ model vào RAM cùng lúc).

**Kỹ năng cần có:**
- Biết dùng `torch.onnx.export`, `coremltools`, `tf2onnx`, `ai_edge_torch`
- Hiểu **shape inference** và **dynamic shapes** — nhiều conversion tool yêu cầu fixed input shape
- Biết trace model đúng cách (`torch.jit.trace` vs `torch.export`) và biết khi nào cái nào fail
- Kiểm tra numerical equivalence sau conversion — output của converted model phải khớp với original trong tolerance chấp nhận được

**Lỗi hay gặp:** trace với wrong input shape, export model đang ở training mode thay vì eval mode, không handle stateful models (RNN, streaming ASR).

---

## 3. Đánh Giá Khả Thi Deploy Trên Mobile

Trước khi bắt đầu convert hay optimize, cần trả lời câu hỏi: **model này có deploy được không?**

**Các chỉ số cần đánh giá:**

| Chỉ số | Ngưỡng thực tế (mid-range device) |
|--------|----------------------------------|
| Parameters | < 10M cho real-time, < 100M cho batch |
| Peak RAM | < 150MB (Android), < 200MB (iOS) |
| Model size trên disk | < 20MB lý tưởng, < 100MB chấp nhận được |
| Latency (inference) | < 100ms cho interactive, < 500ms cho background |
| FLOPs | < 300M FLOPs/inference cho real-time camera |

**Nhưng lý thuyết khác xa thực tế.** Peak RAM khi chạy thực tế có thể cao hơn 2-3x so với model size do activation buffers. FLOPs không phản ánh memory bandwidth bottleneck — và đây mới thường là bottleneck thật sự, đặc biệt với LLM.

Một điểm đáng chú ý: nếu bạn đang evaluate một LLM dựa trên kiến trúc **BitNet 1.58** (weights ternary {-1, 0, +1}), toàn bộ cách tính memory và compute thay đổi. Matrix multiplication trở thành phép cộng/trừ thuần túy, memory bandwidth giảm 8-16x so với FP16. Các ngưỡng bảng trên không áp dụng được — cần benchmark riêng trên device thực.

**Thermal throttling** là vấn đề thường bị bỏ qua: device tầm trung có thể drop performance 40-60% sau 30 giây chạy liên tục do nhiệt. Cần test sustained performance, không chỉ burst.

**Benchmark trên đúng device tier.** Luôn test trên "p80 device" — thiết bị ở percentile thứ 80 của user base của bạn, không phải flagship. Một model chạy ngon trên Pixel 9 có thể không dùng được trên Redmi Note 12.

---

## 4. Fine-tune và Quantize Model

Đây là hai công cụ chính để bridge gap giữa model research và model production.

**Quantization — từ lý thuyết đến thực chiến:**

Post-Training Quantization (PTQ) là điểm xuất phát: nhanh, không cần retrain, thường mất 1-3% accuracy. Quantization-Aware Training (QAT) tốn công hơn nhưng recover accuracy tốt hơn bằng cách insert fake quantization nodes vào training graph. INT8 là standard; INT4 đang nổi lên đặc biệt cho LLM.

Với LLM, **GGUF format của llama.cpp** là cách tiếp cận PTQ thực tế nhất hiện tại. Thay vì một threshold duy nhất, GGUF cung cấp spectrum quantization levels để lựa chọn theo constraint của device:

| GGUF Level | Bit/weight | Dùng khi |
|------------|-----------|---------|
| Q2_K | ~2.6 bit | RAM cực kỳ hạn chế, chấp nhận quality thấp |
| Q4_K_M | ~4.5 bit | Sweet spot — balance tốt nhất giữa size và quality |
| Q5_K_M | ~5.5 bit | Khi RAM còn dư, muốn quality tốt hơn |
| Q8_0 | ~8 bit | Gần như lossless, dùng để debug hoặc benchmark |

Biết chọn đúng level cho từng device tier là kỹ năng thực chiến, không có trong sách giáo khoa.

**Nhưng quantization có giới hạn.** Tất cả các approach trên đều bắt đầu từ một model FP32 đã train xong, rồi giảm precision sau. **BitNet b1.58** (Microsoft Research) đặt câu hỏi khác hẳn: *tại sao không train ngay từ đầu với weights ternary {-1, 0, +1}?*

Kết quả: không còn phép nhân trong matrix multiplication — chỉ cộng và trừ. Memory bandwidth giảm 8-16x. Energy consumption giảm đáng kể. Đây không phải là quantize model đã train, mà là một training paradigm hoàn toàn khác, với inference runtime riêng (`bitnet.cpp`). Llama.cpp cũng đã hỗ trợ BitNet qua format `-IQ1_S` để benchmark so sánh.

BitNet 1.58 hiện vẫn ở giai đoạn early adoption — model ecosystem chưa rộng bằng GGUF. Nhưng nếu bạn đang build pipeline cho LLM on-device dài hạn, đây là hướng cần theo dõi sát.

**Fine-tuning cho on-device:**
- Fine-tune trên representative data của target domain trước khi quantize — giúp model "cứng" hơn với quantization noise
- **LoRA / QLoRA** cho LLM: giảm memory footprint khi fine-tune drastically
- Luôn có calibration dataset tốt cho PTQ — data phải represent đúng real-world input distribution

---

## 5. Accelerator Targeting — Kỹ Năng Phân Biệt Engineer Giỏi

Đây là điểm mà hầu hết engineer bỏ qua, nhưng tạo ra impact lớn nhất.

Mỗi platform có dedicated AI accelerator:
- **iOS** → Apple Neural Engine (ANE) — cực kỳ nhanh, nhưng cần model đúng format và đúng op
- **Android Qualcomm** → Hexagon DSP / Qualcomm AI Engine
- **Android MediaTek** → APU (AI Processing Unit)
- **Android Samsung Exynos** → NPU

**Vấn đề:** không phải mọi layer đều chạy được trên NPU. Khi một op không được support, nó fallback về CPU — và việc copy data giữa CPU và NPU (memory transfer overhead) đôi khi còn tệ hơn chạy hết trên CPU.

Cùng một model: CPU → 150ms, GPU → 40ms, NPU → 8ms. Đây là lý do tại sao accelerator targeting quan trọng đến vậy.

**llama.cpp** là ví dụ tốt nhất về cách một runtime xử lý multi-accelerator targeting: cùng một codebase, build với `LLAMA_METAL=1` cho Apple Silicon (ANE + GPU), `LLAMA_VULKAN=1` cho Android GPU, `LLAMA_BLAS=1` cho CPU-optimized path. Đây là pattern đáng học vì nó giải quyết đúng bài toán: một model, nhiều hardware target, không cần viết lại inference logic.

**Kỹ năng cần có:**
- Biết kiểm tra delegate compatibility trước khi build (TFLite Delegate API, CoreML model validation)
- Hiểu cách partition model: phần nào chạy trên NPU, phần nào fallback
- Biết reshape model architecture để maximize NPU utilization (tránh dynamic shapes, tránh ops không được hỗ trợ)

---

## 6. Operator Compatibility — Biết Trước Để Không Hối Hận Sau

Không phải tất cả PyTorch/TF operators đều được support trên mọi runtime. Phát hiện điều này sau khi đã optimize model là tốn kém.

**Ví dụ thực tế:**
- TFLite không hỗ trợ `tf.math.top_k` với dynamic k
- CoreML không hỗ trợ một số custom attention variants
- Nhiều runtime không hỗ trợ control flow phức tạp (if/while trong model graph)

**Cách làm đúng:**
1. Check operator support list của target runtime **ngay từ đầu**
2. Dùng TFLite Model Analyzer, CoreML Validator, hay ONNX checker để detect unsupported ops **trước khi** đưa cho data scientist design model
3. Nếu có thể, tham gia vào giai đoạn architecture design để tránh dùng ops không tương thích

**Lý tưởng nhất:** AI Engineer on-device nên có mặt trong design review của model, không chỉ nhận model khi đã xong.

---

## 7. Device-Native Profiling — Đo Thực Tế, Không Tin Lý Thuyết

FLOPs và parameter count là ước tính lý thuyết. Latency thực sự cần được đo trên device thực.

**Toolchain theo platform:**

**Android:**
- **Android GPU Inspector** — profile GPU/NPU execution, xem bottleneck ở layer nào
- **Snapdragon Profiler** — dành riêng cho Qualcomm, xem DSP utilization
- **Android Studio Profiler** — RAM, CPU, battery trong khi inference chạy
- **systrace / perfetto** — low-level timing analysis

**iOS:**
- **Xcode Instruments → Core ML Instrument** — xem từng layer chạy bao lâu, chạy ở đâu (CPU/GPU/ANE)
- **Metal Performance HUD** — GPU frame timing
- **Energy Log** — battery impact

**LLM on-device:** các tool trên đo per-inference latency. Với LLM, metric quan trọng hơn là **tokens/second** — tức throughput của decode loop. `llama-bench` (built-in trong llama.cpp) là tool chuẩn để đo chỉ số này trực tiếp trên device, với output rõ ràng theo từng quantization level và backend.

**Những gì cần đo:**
- **Latency** — time to first result, P50/P95 (không chỉ average)
- **Sustained throughput** — performance sau 30 giây, 2 phút liên tục
- **Memory watermark** — peak RAM trong suốt inference cycle
- **Battery drain** — inference loop chạy 10 phút tốn bao nhiêu % pin
- **Thermal behavior** — device có throttle không, throttle lúc nào

---

## 8. Pruning và Knowledge Distillation — Ngoài Quantization

Quantization giảm precision. Nhưng đôi khi bạn cần giảm **cả model architecture** để đạt target latency.

**Structured Pruning:**
- Loại bỏ toàn bộ channels/heads/layers thay vì individual weights
- Structured pruning → model nhỏ hơn thực sự → inference nhanh hơn trên hardware
- Unstructured pruning (sparse weights) trông đẹp trên paper nhưng ít hardware nào khai thác được sparsity hiệu quả

**Knowledge Distillation:**
- Train một student model nhỏ hơn học từ output của teacher model lớn hơn
- Kết quả thường tốt hơn train student từ đầu với cùng size
- Đặc biệt hiệu quả khi kết hợp: distillation → pruning → quantization

**Thứ tự tối ưu điển hình:**
```
Teacher model (large)
    ↓ Knowledge Distillation
Student model (smaller architecture)
    ↓ Fine-tune on domain data
    ↓ Structured Pruning
    ↓ QAT (Quantization-Aware Training)
    ↓ Convert sang inference format
    ↓ Deploy
```

---

## 9. LLM On-Device — Mảng Đang Bùng Nổ

Với sự xuất hiện của Phi-3-mini, Gemma, Llama-3.2, on-device LLM đang trở thành production reality. Đây là skillset riêng mà AI Engineer on-device cần nắm — và nó kéo theo hầu hết mọi kỹ năng đã đề cập ở trên, nhưng với những thách thức đặc thù hơn.

**Thách thức đặc thù của LLM:**
- **KV Cache management** — cache key/value attention states để tránh recompute, nhưng cache size tăng linear theo context length, ăn RAM rất nhanh
- **Prefill vs Decode latency** — prefill (xử lý prompt) và decode (generate từng token) có bottleneck khác nhau, cần optimize riêng
- **Speculative decoding** — dùng draft model nhỏ để predict tokens, model lớn verify — tăng throughput 2-3x; llama.cpp hỗ trợ natively
- **Context window trade-off** — context dài hơn = tốn RAM hơn, latency cao hơn; cần chọn đúng cho use case
- **Streaming output** — user experience tốt hơn khi stream từng token, cần thiết kế pipeline phù hợp

Khi làm việc với LLM on-device, **llama.cpp** và **bitnet.cpp** là hai runtime cần nắm trước tiên — không phải vì chúng là duy nhất, mà vì chúng là reference implementation mà hầu hết framework khác đều học theo. Hiểu cách llama.cpp xử lý quantization, backend delegation, và KV cache là nền tảng để đánh giá bất kỳ LLM runtime nào khác.

**Ecosystem hiện tại:**
- **llama.cpp** — runtime chính cho GGUF, CPU + GPU + NPU, cross-platform
- **bitnet.cpp** — runtime chuyên biệt cho BitNet 1.58 models
- **ExecuTorch** — PyTorch's official mobile LLM runtime, tích hợp tốt với Android/iOS native
- **MLC LLM** — compile LLM cho từng hardware target với TVM backend
- **MediaPipe LLM Inference API** — Google's high-level API, ít control hơn nhưng dễ integrate

---

## 10. Native SDK Integration

Model tốt mà không integrate được vào app thì vô nghĩa. AI Engineer on-device cần biết viết glue code.

**Android:**
- JNI / NDK — gọi C++ inference engine (như llama.cpp) từ Kotlin/Java
- TFLite Java/Kotlin API, ONNX Runtime Android API
- CameraX + ML Kit integration cho real-time camera pipeline

**iOS:**
- CoreML Swift API — load model, run prediction, handle async inference
- Vision framework — wrap CoreML model cho image tasks
- AVFoundation — camera/audio pipeline cho real-time inference

**Những gì cần biết:**
- Threading model: inference thường nên chạy trên background thread, không block main thread
- Memory management: đặc biệt quan trọng trên iOS với ARC
- Error handling: model load fail, inference timeout, low-memory warning

---

## 11. Model Update Pipeline — OTA Không Cần Release App

Một kỹ năng thường bị bỏ qua nhưng cực kỳ quan trọng trong production: **làm thế nào để update model mà không cần release app mới?**

**Vấn đề:** App review cycle (iOS App Store) mất 1-7 ngày. Nếu mỗi lần cải thiện model đều phải release app, iteration cycle rất chậm.

**Giải pháp:**
- **Remote model serving** — host model trên CDN, download và cache trên device
- **Model versioning** — version model độc lập với app version
- **Incremental updates** — chỉ download phần thay đổi (delta update) thay vì full model
- **Rollback mechanism** — khi model mới có vấn đề, fallback về model cũ
- **A/B testing** — ship model mới cho 10% user trước, đo metrics, rồi rollout toàn bộ

**Firebase ML, AWS ML Models, hay custom CDN** đều là lựa chọn phổ biến cho pattern này.

---

## 12. Testing và Validation Pipeline

Sau mỗi bước optimization (quantize, prune, convert), cần validate rằng model vẫn đúng.

**Accuracy regression testing:**
- So sánh output của original model vs converted/quantized model trên test set
- Đặt threshold chấp nhận được: ví dụ accuracy drop < 1%, cosine similarity > 0.99
- Automated test sau mỗi conversion step trong CI pipeline

**Latency benchmarking:**
- Benchmark trên nhiều device tiers, không chỉ một device
- Chạy warm-up iterations trước khi measure (JIT compilation, cache warmup)
- Measure P50, P95, P99 — không chỉ average

**Integration testing:**
- Test toàn bộ pipeline: camera frame → preprocess → inference → postprocess → UI update
- Test edge cases: low-light input, unusual aspect ratios, out-of-distribution input

---

## Tổng Kết: Roadmap Thực Tế

```
Tháng 1-2: Foundation
├── Runtime selection & setup (TFLite, CoreML, ONNX RT)
├── Model conversion pipeline (PyTorch/TF/HF → mobile format, GGUF)
└── Basic quantization (PTQ INT8, GGUF Q4_K_M)

Tháng 3-4: Optimization
├── Accelerator targeting (NPU delegate, llama.cpp backends)
├── Profiling với device-native tools + llama-bench
├── Operator compatibility check workflow
└── Knowledge distillation & structured pruning

Tháng 5-6: Production
├── Native SDK integration (JNI/NDK, CoreML Swift)
├── Model update pipeline (OTA)
├── Testing & validation automation
└── LLM on-device: KV cache, speculative decoding, BitNet 1.58

Ongoing
└── Track hardware releases, new runtime versions, emerging architectures
```

**Kỹ năng tạo ra impact lớn nhất, ít người làm tốt nhất:** accelerator targeting + device-native profiling. Nắm hai cái này vững là bạn đã ở top 10% của field.

---

*On-device AI là một trong số ít ngành mà bạn cần hiểu cả ML lẫn systems lẫn hardware để làm tốt. Đó cũng chính là lý do tại sao người làm giỏi hiếm — và được trả rất cao.*