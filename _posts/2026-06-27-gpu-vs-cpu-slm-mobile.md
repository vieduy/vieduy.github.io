---
title: "GPU Isn't Always the Answer: Benchmarking CPU vs GPU for Small Language Models on Mobile"
date: 2026-06-27 12:00:00 +0700
categories: [AI, SLM]
tags: [on-device, benchmark, litert, gemma, mobile]
toc: true
math: true
---

*A measured look at where the GPU wins, where it loses, and the one ratio that decides it — running gemma-3-270m on an iPhone XS.*

---

## TL;DR

I deployed a 270M-parameter LLM on an iPhone XS and benchmarked the **CPU (XNNPACK)** against the **GPU (Metal)** backend with LiteRT-LM. The result surprised me:

- **Decode (token generation): CPU wins by 2.1×** (47.5 vs 22.5 tok/s).
- **Prefill (prompt processing): GPU wins by up to 2.1×**, and its lead *widens* with prompt length.
- The two cancel out differently depending on workload. The GPU only wins **overall** when your **prompt-to-output token ratio is above ~15:1**.

So for a chatbot (short prompt, long answer), the CPU is faster *and* simpler. For summarization or RAG (long context, short answer), the GPU pulls ahead. "Use the GPU, it's faster" is **not** a safe default for on-device SLMs.

All numbers below are from the engine's own instrumentation, 4 repeats each, with <3% variance.

---

## The setup

| | |
|---|---|
| Model | gemma-3-270m (instruction-tuned), ~270M params, int8/int4 quant |
| Runtime | LiteRT-LM (Google AI Edge) v0.12.0 |
| Device | iPhone XS — Apple A12 Bionic, unified LPDDR4X memory |
| Backends | CPU = XNNPACK · GPU = Metal delegate |
| Context | 1024 tokens |

The app is a tiny translator with a benchmark mode that measures throughput, latency, and peak RAM, plus a CPU/GPU toggle. Everything is timed with **LiteRT-LM's built-in `benchmark_info`**, which reports prefill and decode throughput *separately* — that distinction turns out to be the whole story.

---

## Plot twist #1: the GPU produced garbage

Before I could benchmark anything, the GPU backend generated nothing but `<pad>` tokens — fast, fluent garbage. The first error in the logs pointed at a missing sampler library, which sent me down a rabbit hole. It was a **red herring**: the engine fell back to a static sampler just fine.

The real cause, confirmed from device logs: **fp16 activation overflow**. Gemma-2/3 have large activation magnitudes that overflow half-precision (this is well documented — Gemma is supposed to run in bf16/fp32, never fp16). The iOS GPU path defaults the decoder to **fp16**, so Gemma's logits became `NaN`, and every sampled token collapsed to token 0 (`<pad>`). The CPU path runs fp32 internally, which is why it was fine all along.

**The fix** was one line — force F32 activations when the backend is GPU:

```cpp
if (backend == "gpu") {
    litert_lm_engine_settings_set_activation_data_type(settings, /*F32=*/0);
}
```

After that, `Invalid decode` warnings went from 768 to **0**, and Gemma produced correct text on the GPU. A useful aside: **Qwen2.5 ran on the GPU unmodified** — it's fp16-safe. So if you're picking an SLM specifically to run on a mobile GPU, model choice matters: not every architecture survives fp16.

(Lesson for the debugging section of your brain: don't trust the first scary log line, and force-test your hypothesis. I initially blamed quantization and the sampler — both wrong, both disproven by measurement.)

---

## Plot twist #2: fixed... and still slower

With correct output on both backends, I expected the GPU to pull ahead. It didn't:

```
Decode throughput:  CPU 47.5 tok/s   |   GPU 22.5 tok/s
```

The GPU was **2.1× slower**. To understand why, you have to split LLM inference into its two very different phases.

---

## The key idea: prefill and decode are opposite workloads

An autoregressive LLM does two things:

1. **Prefill** — process the entire prompt *in parallel* to populate the KV cache. This is a matrix × **matrix** multiply over all prompt tokens at once. Lots of math per byte of weights loaded → **high arithmetic intensity** → compute-bound.

2. **Decode** — generate output tokens **one at a time**. Each step is a matrix × **vector** multiply (a GEMV) that must stream *all* the model's weights but does almost no math with them. **Low arithmetic intensity** → bound by memory and per-step overhead.

GPUs are throughput machines: thousands of ALUs that shine when there's a mountain of parallel math (prefill). They're poorly suited to batch-1, low-intensity work (decode), where most of those ALUs sit idle waiting on memory.

And there's a **mobile-specific twist**: on a desktop, a discrete GPU has its own VRAM with ~10–20× the CPU's memory bandwidth, so it wins even on memory-bound work. On a phone, the **CPU and GPU share the same unified memory** — so for decode, the GPU has *no bandwidth advantage*, only extra per-token dispatch and sync overhead.

Let's prove it.

---

## The experiment: sweep prompt length, measure both phases

I swept prompt length from 24 to 656 tokens, 4 repeats each, recording the engine's own prefill and decode throughput.

### Prefill — GPU wins, and the lead grows

| Prompt tokens | CPU prefill (tok/s) | GPU prefill (tok/s) | GPU advantage |
|---:|---:|---:|---:|
| 24 | 77.5 ± 1.3 | 129.9 ± 3.5 | 1.68× |
| 48 | 150.8 ± 1.5 | 263.4 ± 2.8 | 1.75× |
| 144 | 223.8 ± 1.5 | 423.8 ± 2.2 | 1.89× |
| 336 | 338.8 ± 2.1 | 677.8 ± 1.7 | 2.00× |
| 656 | 321.7 ± 3.5 | 666.2 ± 6.8 | 2.07× |

This is the textbook **compute-bound** signature: the more parallel work (longer prompt), the further the GPU pulls ahead — 1.68× → 2.07× — until both saturate (CPU plateaus near 322 tok/s, GPU near 666). The GPU's ceiling is ~2× higher.

### Decode — CPU wins, flat as a board

| | CPU | GPU |
|---|---:|---:|
| Decode (tok/s) | **47.5 ± 0.4** | 22.5 ± 0.5 |

Decode throughput is **independent of prompt length** (it's per-token) and the CPU wins **2.1×** at every point. The GPU's parallelism is wasted, and its dispatch overhead per token makes it worse.

> **An honest caveat for the methodically minded:** at 47.5 tok/s the CPU is moving roughly 13 GB/s of weights — only ~38% of the A12's ~34 GB/s peak. So decode is **not** saturating the memory bus. The accurate framing is *low arithmetic intensity + per-token overhead*, not "bandwidth-bound." I also expected forcing F32 (vs fp16) to halve GPU decode; it didn't (21.9 vs 22.8 tok/s) — because the **weights** dominate decode traffic, not the activations. Measure, don't assume.

---

## The one number that decides it: the prompt/output ratio

A real request costs `prefill_time + decode_time`. Plugging in the measured rates:

- The GPU **saves ~1.6 ms per prompt token** (faster prefill).
- The GPU **costs ~23.4 ms per output token** (slower decode).

Set GPU total < CPU total and solve:

$$\frac{\text{prompt tokens}}{\text{output tokens}} > \frac{23.4}{1.6} \approx \mathbf{14.5}$$

You need **~15+ prompt tokens for every output token** before the GPU's prefill win covers its decode penalty.

| Workload | prompt : output | Faster backend |
|---|---|---|
| Chatbot / assistant | ~1 : 5 | **CPU** (decisively) |
| Translation | ~1 : 2 | **CPU** |
| Summarization (long doc → short summary) | ~20 : 1 | **GPU** |
| Classification / extraction (big context → label) | ~50 : 1 | **GPU** |
| RAG answer over retrieved chunks | 10–50 : 1 | **usually GPU** |

---

## A decision guide for shipping SLMs on mobile

- **Default to CPU for conversational / generation-heavy apps.** It's faster for decode-dominated workloads *and* simpler (no fp16 pitfalls, lower peak RAM — 142 MB vs 229 MB here).
- **Reach for the GPU when prefill dominates:** long context, short output — summarization, classification, RAG, reranking, scoring.
- **Validate output, not just speed.** A backend can be fast and wrong (`<pad>`). Gemma needed F32 on the GPU; Qwen2.5 didn't. Always check correctness per backend.
- **Watch precision.** fp16 is the GPU's fast path, but some architectures (Gemma) overflow it. If your GPU output is garbage, suspect precision before quantization.
- **Consider thermals and contention** (not benchmarked here): sustained GPU use competes with the UI/display and heats the device; the CPU may throttle differently. Measure under your real duty cycle.

---

## Methodology & honesty notes

- Numbers are LiteRT-LM's own `benchmark_info` (prefill/decode tok/s measured separately), 4 repeats, mean ± std shown; variance < 3%.
- Single device (iPhone XS / A12), single model family, one runtime version. **The *shape* of the result generalizes** (prefill→GPU, decode→CPU on unified-memory mobile SoCs); **the exact crossover (~14.5) does not** — it will shift with model size, quantization, chip, and runtime.
- Decode is not memory-bandwidth-*saturated* here; it's overhead- and intensity-limited. Don't overstate the roofline.
- Theory references: the **roofline model** (Williams, Waterman, Patterson) and arithmetic intensity are the standard lens for this; nothing here is novel physics, just measured on a phone.

---

## Conclusion

For small language models on mobile, **the GPU is not a free win.** It's genuinely better at *prefill* — up to ~2× here — but loses *decode* by a similar margin, and most consumer LLM workloads are decode-heavy. The right backend is a function of your **prompt-to-output ratio**, not a default.

If you take one thing away: **benchmark prefill and decode separately, on your real workload, and check correctness per backend.** The averaged "tokens per second" on a spec sheet hides the entire decision.

---

*Reproducibility: gemma-3-270m, LiteRT-LM v0.12.0, iPhone XS (A12). Throughput from LiteRT-LM `benchmark_info`. Happy to share the benchmark harness.*
