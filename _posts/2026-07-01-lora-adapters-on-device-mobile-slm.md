---
title: "One Small Model, Many Skills: Switching LoRA Adapters On-Device to Get the Most Out of a Mobile SLM"
date: 2026-07-01 10:00:00 +0700
categories: [AI, SLM]
tags: [on-device, llama-cpp, lora, adapters, gemma, ios, swift, mobile, tutorial]
toc: true
---

*A small on-device model can't be great at everything — but it can **become** the right specialist on demand. This post ships one frozen base model plus a library of small LoRA adapters, then loads, switches, and frees them at runtime — so a single mobile SLM covers many tasks and languages without reloading the model or shipping N copies of it. It covers the runtime mechanics, the download-and-hot-load flow, and the memory/latency numbers that decide your switching strategy.*

---

## What you'll build

A SwiftUI translator app that runs **Gemma-3-270m** on-device via **llama.cpp**, and can:

- **hot-load a LoRA adapter at runtime** — including one *downloaded after the app launched*,
- **swap adapters with zero downtime** (no base-model reload),
- **free the unused adapter** to keep memory flat, and
- report the exact **RAM and switch latency** so you can pick a residency policy on real numbers.

This is the "adapters" sequel to my [1-bit llama.cpp on iOS post]({% post_url 2026-06-29-llamacpp-bonsai-1bit-llm-ios %}) — same app shell, now with dynamic fine-tunes.

**Audience:** AI / mobile engineers. No prior LoRA or Metal experience required.

---

## Why one small model isn't enough

On-device LLMs earn their place for concrete engineering reasons — **privacy** (nothing leaves the device), **offline** operation, **zero per-call cost**, and **low, predictable latency**. The price of admission is size: to fit a phone's memory and thermal envelope you run a *small* model. Gemma-3-270m is 270M parameters, ~278 MB. That's the whole tension of this post — a small model is what makes on-device viable, and also what makes it limited.

**The limit is capacity, and it bites hardest exactly where you'd want multi-task.** A 270m model has little spare representational room. Push it hard toward one task with full fine-tuning and it overwrites what it knew — *catastrophic forgetting* — and the effect is measurably **worse on small models**. Recent studies find smaller models "suffer sharp performance drops on previous tasks once new tasks are introduced," and that forgetting "persists when the model has little remaining capacity … models pretrained close to saturation cannot absorb new information without overwriting prior knowledge" ([Kotha et al.](https://arxiv.org/pdf/2504.01241); [continual-learning on-device study](https://arxiv.org/pdf/2602.00166)). So fully fine-tuning a small model into a great EN↔VI translator tends to make it *worse* at summarization, JSON, and every other language pair. Do it again for the next task and you have a *different* model, not an upgraded one.

The naive fix — **one fully fine-tuned model per task** — avoids forgetting but is unshippable on a phone, because it's the same frozen base duplicated N times:

```
   N full fine-tuned models                 One base + N LoRA adapters
   (avoids forgetting, unshippable)         (this post)

   ┌───────────┐ ┌───────────┐              ┌──────────────────────────────┐
   │ base 278M │ │ base 278M │              │          base 278 MB         │
   │ + EN↔VI   │ │ + EN↔JA   │              │   (frozen · shared · bundled)│
   └───────────┘ └───────────┘              └──┬──────┬──────┬──────┬───────┘
   ┌───────────┐ ┌───────────┐                 │29 MB │29 MB │29 MB │  ...
   │ base 278M │ │ base 278M │                 │ EN↔VI│ EN↔JA│ summ │
   │ + summ    │ │ + JSON    │                 └──────┴──────┴──────┘
   └───────────┘ └───────────┘                   ↑ downloaded on demand

   5 tasks ≈ 1.4 GB                          278 + 5×29 ≈ 423 MB
   (~95% duplicate base weights)             (one base, reused)
```

Five tasks ≈ 1.4 GB, ~95% of which is *identical* base weights — and that assumes you can even store them. On-device, "storing multiple task-specific [models] quickly exhausts local memory" ([2602.00166](https://arxiv.org/pdf/2602.00166)); we hit exactly this tension later when choosing how many adapters to keep in RAM.

**LoRA breaks the coupling between base and task.** Instead of rewriting the model, it freezes the base and learns a **low-rank delta** for each task, confined to a small subspace of the weights (~29 MB here at rank 64, ~10% of the base). Because updates are restricted to that subspace, LoRA "acts as a regularizer, preserving the base model's capabilities" and mitigates the forgetting that full fine-tuning causes ([LoRA paper](https://arxiv.org/pdf/2106.09685)). The base's general competence stays intact; the adapter adds one focused skill on top. It's the same mechanism used to *add languages* to multilingual models without retraining or "rehearsing" the old ones ([LoRA multilingual ASR](https://arxiv.org/pdf/2408.10680)).

One subtlety matters specifically for multi-task: naively **merging** several LoRAs into one weight set reintroduces **cross-task interference** ([LoRI](https://arxiv.org/pdf/2504.07448)). We sidestep it entirely — we never merge. At any instant exactly **one** adapter is *activated* on the live model, so each task runs against a clean base + its own delta. Multi-task becomes a matter of *which* delta is currently attached — which is the runtime problem the rest of this post solves.

**Honest ceiling:** an adapter **steers** the base, it doesn't **enlarge** it. LoRA teaches a 270m model the *shape* of a task — the format of a translation, the register of a summary, the grammar of a language pair — but it can't add knowledge or reasoning the base fundamentally lacks. For well-scoped tasks like translation that's exactly the right tool; for tasks needing facts a 270m brain never had, you need a bigger base, not a bigger adapter.

That reframes deployment:

- **Ship the base once.** Bundle the 278 MB base in the app.
- **Stream expertise on demand.** Download a 29 MB `.gguf` per task/language — not a full model.
- **Update without the App Store.** Push an improved adapter from your server; the app picks it up at runtime.
- **Switch specialists instantly.** One resident base, many hot-swappable experts.

All of it depends on one capability most on-device stacks lack: attaching an adapter to a **live** model — download, apply, swap, free, without reloading the base. Let's see which runtime can.

---

## Why llama.cpp (and not LiteRT-LM)

I started on LiteRT-LM. Its text-LoRA path is **stubbed** in the build I had — the engine returns *"Lora is not supported."* for text. llama.cpp, by contrast, has a **mainline, working LoRA C API**, and — usefully — my prebuilt `llama.xcframework` already exported it. No rebuild required:

```c
// llama.h — the three calls that make this whole post possible
struct llama_adapter_lora * llama_adapter_lora_init(struct llama_model *, const char * path_lora);
int32_t                     llama_set_adapters_lora(struct llama_context *, llama_adapter_lora **, size_t, float * scales);
void                        llama_adapter_lora_free(struct llama_adapter_lora *);
```

Verify your framework exports them before writing a line of Swift:

```bash
nm -gU Frameworks/llama.xcframework/ios-arm64/llama.framework/llama | grep -i adapter_lora
```

---

## How LoRA hot-swap actually works

The reason swapping is instant is that llama.cpp **never modifies the base weights.**

A LoRA for a weight `W` isn't a new copy of `W` — it's a low-rank pair `(A, B)`. The adapted weight is conceptually:

```
W_effective = W + scale · (B · A)      // B·A is rank-64 here → tiny
```

llama.cpp keeps `W` frozen and applies the correction **at compute time**, splitting the work into two phases:

| Phase | Call | Cost | When |
|---|---|---|---|
| **Load** | `llama_adapter_lora_init` | read ~29 MB, allocate a buffer, bind tensors to the base | ahead of time / on download |
| **Activate** | `llama_set_adapters_lora` | write pointers + scales onto the `llama_context` | the swap — **O(1)** |

When `llama_decode` builds the graph for the next token, each relevant matmul becomes `y = W·x + scale·(B·(A·x))` (the `build_lora_mm` path in ggml). So the adapter is applied as a few extra small matmuls fused into the graph — recomputed each forward pass from whatever adapters are currently set on the context.

That's why the swap is free: **no merge, no reload, no re-tokenize.** Switching adapters just changes which small `A`/`B` tensors the *next* graph reads. The only cost is deferred to compute (a small per-token overhead), never to the swap itself. The one rule: swap **between** generations, not mid-decode, or the KV cache (computed with the old adapter) becomes inconsistent.

---

## The bridge: three methods

The Obj-C++ bridge over `llama.h` needs exactly three additions. All operate on the model/context that are already resident for the app's lifetime.

```objc
// Load once, keyed by identifier. Can be called at ANY runtime moment.
- (BOOL)loadAdapterAtPath:(NSString *)path identifier:(NSString *)id {
    llama_adapter_lora *a = llama_adapter_lora_init(_model, path.UTF8String);
    if (!a) { _lastError = "init failed (arch/tokenizer mismatch?)"; return NO; }
    if (_adapters.count(id)) llama_adapter_lora_free(_adapters[id]);  // replace
    _adapters[id] = a;                                               // std::unordered_map
    return YES;
}

// Activate on the live context — the zero-downtime swap. nil id = pure base.
- (BOOL)setActiveAdapter:(NSString *)id scale:(float)scale {
    if (id.length == 0) { llama_set_adapters_lora(_ctx, nullptr, 0, nullptr); return YES; }
    auto it = _adapters.find(id.UTF8String);
    if (it == _adapters.end()) return NO;
    llama_adapter_lora *a[1] = { it->second };
    float s[1] = { scale };
    llama_set_adapters_lora(_ctx, a, 1, s);
    return YES;
}

// Free the ~29 MB buffer (detach from ctx first). Enables load-on-demand policies.
- (BOOL)unloadAdapter:(NSString *)id {
    auto it = _adapters.find(id.UTF8String);
    if (it == _adapters.end()) return NO;
    llama_set_adapters_lora(_ctx, nullptr, 0, nullptr);
    llama_adapter_lora_free(it->second);
    _adapters.erase(it);
    return YES;
}
```

Expose them through your engine protocol with **no-op defaults** so a second (LiteRT) engine still conforms without implementing them.

---

## The killer feature: download an adapter and apply it at runtime

This is the part people don't believe works until they see it. `llama_adapter_lora_init` takes the **live model** and a **filesystem path** — so a file that arrives *after* launch works exactly like a bundled one:

```
user taps "Get task 2"
  1. download   →  URLSession/copy the .gguf into Documents/     (writable; the bundle is read-only)
  2. loadAdapter(path: documentsURL)  →  llama_adapter_lora_init(runningModel, path)
  3. setActiveAdapter(...)            →  llama_set_adapters_lora(ctx, …)
  4. next translate → llama_decode uses it
```

At step 2, llama.cpp opens the GGUF, reads the `A`/`B` tensors into a ~29 MB buffer, and **binds each adapter tensor to the matching base tensor already in RAM** (by name, e.g. `blk.0.attn_q.weight.lora_a`). It never re-reads or modifies the 278 MB of base weights — it just attaches low-rank patches to tensors that are already loaded. No restart, no base reload.

**The one hard requirement:** the file must be a **converted GGUF adapter for the same base architecture**:

```bash
# server-side, once, per adapter — llama.cpp can't load raw PEFT safetensors
python convert_lora_to_gguf.py ./my-peft-adapter --base ./gemma-3-270m-it \
    --outfile adapter.gguf --outtype f16
```

A mismatched or raw-PEFT file makes `llama_adapter_lora_init` return null → your `loadAdapter` returns `NO` → surface an error, no crash.

---

## Getting the most out of the base: one task, one adapter

The whole strategy is to make a fixed 270m base *punch above its weight* by never asking it to be a generalist. Each request runs against the base **plus the one adapter that matches the task** — a translation gets the translation adapter, a summary gets the summary adapter, EN↔JA gets the Japanese pair. Same base, different expert, per request. That's how you turn "a mediocre jack-of-all-trades" into "a competent specialist, on demand."

Deciding *which* adapter to activate is a **routing** decision, and you have three practical options:

- **Explicit** — the user chooses the task (our per-task *Translate 1 / Translate 2* buttons). Simplest, unambiguous, and honest about what the model is doing.
- **Contextual** — the UI feature *is* the task: a "Summarize" button always routes to the summary adapter; a language picker routes to that pair. No inference needed.
- **Inferred** — detect the task or language from the input (a lightweight language-ID heuristic, a tiny classifier, or a quick zero-shot pass through the base itself) and pick the adapter automatically.

The routing logic is yours; the runtime cost of acting on it is a single `setActiveAdapter` call. And the SLM-capacity research from the start of this post is exactly the design principle: **don't stretch one small model across everything — route each request to a narrow expert and keep the base a clean generalist underneath.** That's how you extract maximum ability from minimum size.

There's a bonus lever if a task benefits from two skills at once: `llama_set_adapters_lora` takes an *array* of adapters with per-adapter scales, so you can **blend** (e.g., 0.7 × formal-tone + 0.3 × domain-terms) rather than pick one. Blending resident adapters is still a pointer-level operation — no reload.

---

## Memory: what actually counts

I instrumented the footprint (`task_vm_info.phys_footprint`, the number jetsam uses) around every load. On the iPhone XS with the 278 MB base:

- **1 adapter resident → 119 MB total footprint.**
- **2 adapters resident → 147 MB total footprint.** (+28 MB for the second.)
- Each adapter is therefore **~29 MB in RAM** — matching its file size.

Now the counter-intuitive part. If 1 adapter is 29 MB and the total is 119 MB, the **278 MB base model contributes only ~90 MB** to the footprint. Why? Because the base is loaded with `mmap` — its pages are **clean, file-backed**, and iOS doesn't fully charge those to `phys_footprint` (they can be evicted and re-read). The adapter, on the other hand, is loaded into an **allocated, dirty buffer** (`ggml_backend_alloc` + `read_raw`), which *is* fully charged.

> The small adapter is "heavier" per byte in your footprint than the big base model. Loading a LoRA **does** cost RAM; the base mostly doesn't.

---

## Three residency policies (and the latency that decides between them)

Once you can load and free adapters, you get to choose how many stay in RAM. I measured all three on-device.

**Cold-switch latency (only-active policy):** freeing the old adapter + re-loading the new one from disk = **96 ms**. It's paid once per *cold* switch, before the first token — invisible next to a full translation (hundreds of ms to seconds), but real.

| Policy | RAM for N adapters | Switch latency | Best for |
|---|---|---|---|
| **Keep-all-resident** | N × 29 MB | **~0** (pointer swap) | 2–3 small adapters |
| **LRU (keep K hot)** | K × 29 MB | 0 if hot, 96 ms cold miss | many adapters, frequent reuse |
| **Only-active (1 resident)** | 29 MB flat | 96 ms **every** switch | many/large adapters, tight RAM |

The decision rule, using the real numbers:

- **≤ 3 small adapters:** keep-all-resident. 60–90 MB is nothing on a 4 GB device; swaps are instant.
- **5–6 small adapters:** still fine to keep all (~145–174 MB), or use **LRU(2–3)** to bound memory while keeping common switches instant. Pure only-active is *overkill* here — you'd pay 96 ms per switch to save memory you don't need.
- **Many adapters, large adapters, or tight RAM:** only-active. Flat 29 MB regardless of catalog size; only-active saves `(N−1) × 29 MB`.

Implementation is a `Set<Adapter>` of what's resident plus your eviction rule; "only-active" is just `unloadAdapter(previous)` before `loadAdapter(next)`.

---

## A note on running the base efficiently

Adapters are cheap; the base model's decode speed is what the user actually feels. Getting a small model to run fast on-device — quantization, CPU vs GPU, thread count — is its own topic and is **very device-specific** (the right answer on an older phone differs from a current one). That end of it is orthogonal to adapter switching; I cover it in the [companion post]({% post_url 2026-06-29-llamacpp-bonsai-1bit-llm-ios %}). Rule of thumb: on mobile, CPU with a good int8 kernel usually wins for batch-1 decode, and *fewer, faster* threads often beat *more* threads.

One tip that **is** adapter-specific: when checking whether an adapter is actually doing its job, decode with **greedy/deterministic sampling** (`temperature = 0`). Otherwise the same prompt yields different text each run and you can't separate the adapter's effect from sampling noise. Turn sampling back on for production.

---

## Putting it together: the deployment shape

```
┌─────────────────────────────────────────────┐
│ SwiftUI: per-task buttons, gated on download │
│   ↳ ViewModel: residency policy + RAM/latency│
├─────────────────────────────────────────────┤
│ LlamaBridge (Obj-C++): load / activate / free│
├─────────────────────────────────────────────┤
│ llama.xcframework  +  base model.gguf (bundled)
│ Documents/: adapter_*.gguf (downloaded)       │
└─────────────────────────────────────────────┘
```

- **Bundle** the base `model.gguf` once.
- **Download** task adapters at runtime into `Documents/` (converted server-side to `.gguf`).
- **Route** each request to its adapter (explicit / contextual / inferred), then `setActiveAdapter`.
- **Manage RAM** with a residency policy (keep-all / LRU / only-active) sized to your adapter count.

---

## Gotchas & troubleshooting

- **`llama_adapter_lora_init` returns null** → wrong format (raw PEFT safetensors) or wrong base arch. Convert with `convert_lora_to_gguf.py --base <same base>`; the adapter must match the base you shipped.
- **Adapter loads but output doesn't change** → you loaded it but never `setActiveAdapter`'d it, or you're comparing under sampling — use greedy (`temperature = 0`) to see the real effect.
- **Footprint doesn't move on load** → you're *switching* (pointer swap, 0 cost), not *loading*; or you're watching the mmap'd base, which isn't fully charged. The adapter's ~29 MB *does* count — measure across a `load`, not a `setActive`.
- **Handle invalid after a model reload** → adapters are tied to the model instance; if you recreate the base (e.g., switching backend), re-load the adapters from disk.
- **Downloaded a new adapter but it won't apply** → check it's a `.gguf` converted against the *same* base and that it landed in a writable dir (Documents), not the read-only bundle.

---

## Where to go next

- **LRU residency** with a configurable cap — the sweet spot beyond ~3 adapters.
- **Automatic routing** — a tiny language/task classifier that picks the adapter so the user doesn't have to.
- **Real HTTP download** with checksum/size validation before `loadAdapter`, plus an on-device adapter catalog/updater.
- **Adapter blending** — `llama_set_adapters_lora` takes an array with per-adapter scales, so you can mix fine-tunes at inference time (e.g., tone + domain).

The headline: on modern mobile, **fine-tunes are a download, not a redeploy.** A frozen base plus streamable 29 MB adapters — attached to a live model in ~100 ms and freed just as easily — is a genuinely different deployment model than shipping a monolithic fine-tuned model per task. A small model doesn't have to be everything; it has to be able to *become* the one thing the user needs, on demand.

---

## Related reading

- [Running a 1-bit LLM on-device with llama.cpp (Bonsai on iOS)]({% post_url 2026-06-29-llamacpp-bonsai-1bit-llm-ios %}) — the base app shell this builds on.
- llama.cpp `convert_lora_to_gguf.py` and the `/lora-adapters` server endpoint (the same hot-swap mechanism, server-side).

### Why-it-works references

- Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/pdf/2106.09685) — the method; base frozen, low-rank task delta.
- [*Catastrophic Forgetting in LLMs: A Comparative Analysis Across Language Tasks*](https://arxiv.org/pdf/2504.01241) — forgetting is worse when spare capacity is low.
- [*Continual learning of local models with budget constraints*](https://arxiv.org/pdf/2602.00166) — smaller local models forget more; storing many task adapters exhausts device memory.
- [*LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation*](https://arxiv.org/pdf/2504.07448) — merging LoRAs causes interference (which per-task *switching* avoids).
- [*Rehearsal-Free Multilingual ASR: a LoRA case study on Whisper*](https://arxiv.org/pdf/2408.10680) — adding languages via LoRA without forgetting existing ones.
