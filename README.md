# tiny-vLLM

`tiny-vLLM` is a minimal, educational LLM inference engine inspired by the core architectural concepts of vLLM. It runs locally on Apple Silicon (M2 MacBook Air, 8 GB RAM) and serves as an educational guide to understanding LLM serving systems — with a **multi-page comparative-study dashboard** that benchmarks every optimization live.

It is a readable Python implementation of:
- **Continuous Batching** (the most important vLLM throughput optimization)
- **Request-level KV Cache Management** with left-padding + RoPE position IDs
- **Radix-style Shared-Prefix KV Caching**
- **Chunked Prefill**
- **Speculative Decoding** (draft model + main-model verification)
- **Multiple Scheduler Policies** (FIFO / SJF / Priority)
- **Quantization** (fp32 / fp16 / int8)
- **Live Observability** — throughput, batch-size, queue-depth & memory timeseries, per-request Gantt, KV-cache memory bars
- **Ablation Sweep Suite** — chunk size, batch size, concurrency

---

## Table of Contents
1. [Concepts](#concepts)
2. [System Architecture](#system-architecture)
3. [Repository Structure](#repository-structure)
4. [Features & File Locations](#features--file-locations)
5. [API Reference](#api-reference)
6. [Dashboard Pages](#dashboard-pages)
7. [How to Run](#how-to-run)
8. [Benchmark Results](#benchmark-results)
9. [Scope & Non-Goals](#scope--non-goals)

---

## Concepts

### What is Continuous Batching?
In static batching, a server groups prompts and processes them together. If Request A finishes after 5 tokens but Request B needs 30, Request A's slot sits idle until B finishes. **Continuous batching** processes requests at the *token level*: at each decode step the scheduler refills any free slot from the queue the instant a request finishes.

### What is KV Caching?
During autoregressive generation the model predicts the next token from all previous tokens. Without a KV cache we'd recompute keys/values for the entire history at every step ($O(N^2)$). With **KV caching**, past K/V tensors are stored; each decode step passes only the last token, appends its K/V, and reuses the cache.

**KV Cache in Continuous Batching:** when batching requests with different histories, their caches have different lengths. tiny-vLLM resolves this by **left-padding** shorter caches to the longest, concatenating across the batch, specifying explicit **Position IDs** (so RoPE is correct despite padding), then **slicing/unpadding** after the forward pass.

### Radix-Style Shared-Prefix Cache
Two prompts that share a common head (e.g. the same system prompt) can **reuse the cached K/V for the shared prefix** and prefill only the diverging suffix. tiny-vLLM finds the longest cached token-prefix that is a prefix of a new prompt, clones its KV, and prefills the remainder — turning a full prefill into a short suffix prefill.

### Chunked Prefill
A long prompt's prefill is split into fixed-size **chunks processed one per engine step** (instead of one giant forward pass). This lowers peak memory and lets other active requests decode between chunks, improving TTFT interleaving.

### Speculative Decoding
A small **draft model** autoregressively proposes *K* candidate tokens; the **main model** verifies all *K+1* positions in **one** forward pass. The longest matching prefix is accepted; the first mismatch is corrected by the main model. When the draft is accurate, multiple tokens are produced per main-model step.

---

## System Architecture

```
Client
  ↓ (HTTP Request)
FastAPI Server (server/api.py)
  ↓ (Enqueue)
Scheduler Queue (engine/scheduler.py)
  ↓ (FIFO / SJF / Priority)
Inference Engine (engine/engine.py)
  ↓ (Padded & Concatenated)
KV Cache Manager (engine/kv_cache.py)
  ↓ (Batched Tensors & Position IDs)
Hugging Face Causal LM (engine/model.py)   + Draft Model (speculative)
```

---

## Repository Structure

```
tiny-vllm/
├── engine/
│   ├── model.py        # HF model/tokenizer wrapper (MPS/CPU), precision + quantization, KV byte accounting
│   ├── engine.py       # Core generation loop: prefill/decode, radix cache, chunked prefill, spec decoding, timeseries
│   ├── request.py      # Request dataclass, lifecycle states, timing metrics, Gantt event log, draft KV cache
│   ├── scheduler.py    # FIFO/SJF/Priority scheduler + sequential/static/continuous admission modes
│   ├── kv_cache.py     # Left-padding, batching, and slicing/unpadding KV caches
│   └── sampler.py      # Greedy, temperature, top-k/top-p sampling
│
├── server/
│   └── api.py          # FastAPI routes + /compare + /sweep + /config + /timeseries + /gantt
│
├── benchmarks/
│   └── benchmark.py    # Performance benchmarking script (sequential vs batching)
│
├── dashboard/
│   └── index.html      # Multi-page vanilla HTML/CSS/JS dashboard (Live / Compare / Sweeps / Phases / About)
│
├── docs/
│   └── architecture.md # Deep dive into systems internals and RoPE padding math
│
├── requirements.txt    # Package dependencies
├── main.py             # Entrypoint to run the API & serve the dashboard
└── README.md           # This file
```

---

## Features & File Locations

| Feature | What it does | Location in code |
|---|---|---|
| **Continuous batching** | Token-level slot refilling as requests finish | `engine/engine.py` `_step_inner()`; `engine/scheduler.py` `schedule()` |
| **Sequential / static / continuous modes** | Switchable admission strategies | `engine/scheduler.py` `continuous` flag; `engine/engine.py` `apply_config()` |
| **FIFO / SJF / Priority scheduling** | Queue ordering policies | `engine/scheduler.py` `_pick_next()` |
| **Left-padded batched KV cache** | Pad+concat caches, explicit position IDs, slice/unpad after forward | `engine/kv_cache.py` `pad_and_batch_kv_cache()`, `slice_and_update_kv_cache()` |
| **Radix shared-prefix cache** | Reuse KV across prompts sharing a common prefix | `engine/engine.py` `_find_longest_prefix()`, `_try_prefix_cache()`, `_prefill_suffix()`, `_store_prefix_cache()` |
| **Chunked prefill** | Split long prompts into per-step chunks | `engine/engine.py` `_do_prefill()` (chunked branch) |
| **Speculative decoding** | Draft proposes K tokens, main verifies in one pass, KV rollback | `engine/engine.py` `_decode_speculative()`, `_init_draft_cache()` |
| **Greedy / temperature / top-k / top-p sampling** | Token sampling strategies | `engine/sampler.py` `sample_token()` |
| **Quantization (fp32 / fp16 / int8)** | Weight precision + dynamic int8 quantization | `engine/model.py` `_load_model()`, `reload()` |
| **Hot-swappable main model** | Switch model (360M / 135M / 1.7B) at runtime | `engine/model.py` `reload()`; `engine/engine.py` `apply_config()` |
| **Live throughput timeseries** | Tokens/sec sampled every step | `engine/engine.py` `_record_history()`, `history`; `/timeseries` |
| **Batch-size & queue-depth timeseries** | Per-step decode batch size + queue depth | `engine/engine.py` `_record_history()` |
| **Memory profiler** | RSS + MPS allocated memory over time | `engine/engine.py` `_mem_mb()` |
| **KV-cache memory bars** | Per-active-request KV bytes | `engine/model.py` `kv_bytes_per_token()`; `/status` `kv_bytes` |
| **Per-request Gantt** | Prefill vs decode phase timeline | `engine/request.py` `events`, `record_event()`; `engine/engine.py` `get_gantt()`; `/gantt` |
| **Serving-mode comparison** | Run workload across all 3 modes, fair (no cache) | `server/api.py` `/compare`, `_run_workload()` |
| **Ablation sweeps** | Sweep chunk_size / max_batch_size / concurrency | `server/api.py` `/sweep` |
| **SSE token streaming** | Real-time token stream | `server/api.py` `/stream` |
| **Non-blocking API** | Engine loop in background asyncio task + `to_thread` | `server/api.py` `run_engine_loop()` |
| **Multi-page dashboard** | Live / Compare / Sweeps / Phases / About with hover tooltips | `dashboard/index.html` |
| **Runtime config endpoint** | Change any setting live | `server/api.py` `/config` (GET/POST) |

### Models
- **Main model (default):** `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Speculative draft model:** `HuggingFaceTB/SmolLM2-135M-Instruct` (shares the tokenizer; lazily loaded when spec decoding is enabled)

Both share the SmolLM2 tokenizer, which is required for speculative decoding (the draft's proposed token IDs must be valid for the main model).

---

## API Reference

All routes are served from `server/api.py`.

| Method | Route | Purpose |
|---|---|---|
| POST | `/generate` | Generate text (blocking, returns full text + metrics) |
| GET  | `/stream` | Generate via Server-Sent Events (token-by-token) |
| GET  | `/status` | Active slots, waiting queue, metrics, recent completed (for dashboard) |
| GET  | `/metrics` | Aggregate engine metrics |
| GET  | `/config` | Current engine configuration |
| POST | `/config` | Apply runtime config changes (mode, scheduler, batch, precision, prefix cache, chunked prefill, chunk size, speculative, spec K, model_id) |
| GET  | `/timeseries` | Historical throughput/batch/queue/memory snapshots |
| GET  | `/gantt` | Per-request phase event logs |
| POST | `/compare` | Run workload across sequential/static/continuous, return comparative metrics |
| POST | `/sweep` | Run a parameter sweep (chunk_size / max_batch_size / concurrency) |
| GET  | `/` | Serve the dashboard |

---

## Dashboard Pages

The dashboard (`dashboard/index.html`) is a single-page app with a top nav switching between views. **Every label, metric and chart title has a hover tooltip** (`data-tip`) explaining what it measures.

1. **Live** — full config bar (main model, mode, scheduler, max batch, precision, prefix cache, chunked prefill, chunk size, speculative, spec K), 8 live metric cards, 4 canvas charts (throughput / batch+queue / memory / KV-cache bars), active generation slots, waiting queue, playground with priority field + SSE streaming.
2. **Compare** — serving-modes head-to-head table + bar charts (winners highlighted green), and a dedicated speculative-decoding on/off comparison.
3. **Sweeps** — three one-click ablation panels (chunk size, batch size, concurrency), each renders a results table + bar chart.
4. **Phases** — per-request Gantt (prefill/decode/cache-hit colored), batch timeline, completed request logs.
5. **About** — feature cards explaining every implemented capability.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark (sequential vs batching)
```bash
python benchmarks/benchmark.py
```

### 3. Run the Server & Dashboard
```bash
python main.py
```
Open **http://127.0.0.1:8000/** in your browser. Hard-refresh (`Cmd+Shift+R`) if you previously cached an older dashboard.

> The first launch downloads the SmolLM2-360M model (~720 MB). Enabling speculative decoding additionally loads the 135M draft model.

---

## Benchmark Results

### Serving-Mode Comparison (SmolLM2-360M, 4 concurrent prompts × 12 tokens, prefix cache off for fairness)

| Mode | Total Time | Throughput | Avg TTFT | Avg Latency | Max Batch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **sequential** | 8.68 s | 5.53 t/s | 5.055 s | 6.458 s | 1 |
| **static** | 1.67 s | 28.71 t/s | 0.300 s | 1.667 s | 4 |
| **continuous** | **1.43 s** | **33.64 t/s** | **0.214 s** | **1.381 s** | 4 |

**Continuous batching is ~6× faster than sequential** on throughput — confirming the core vLLM thesis on this hardware.

### Radix Prefix Cache (SmolLM2-360M)
Repeating a prompt drops TTFT from ~2.0 s (cold prefill) to ~0.04 s (exact cache hit, zero forward passes). A prompt sharing a 27-token prefix with a cached prompt reuses those 27 tokens and prefills only the 15-token suffix.

### Speculative Decoding (SmolLM2-360M main + 135M draft)
Functional and correct (KV rollback verified). Throughput gain depends on **draft accuracy**; with the off-the-shelf 135M draft the acceptance rate is low (~10–20 %), so the speedup is modest — itself a legitimate finding: speculative decoding's benefit is governed by draft accuracy, not just the size ratio.

---

## Scope & Non-Goals

### What is implemented
- Full prefill/decode engine loop separation.
- Continuous / static / sequential batching.
- FIFO / SJF / priority scheduling.
- Radix-style shared-prefix KV caching.
- Chunked prefill.
- Speculative decoding (single-request path).
- fp32 / fp16 / int8 quantization with hot-swap.
- Real-time token streaming via SSE.
- Non-blocking concurrent API.
- Live observability + Gantt + ablation sweeps + multi-page dashboard.

### What is intentionally NOT implemented (Non-Goals)
- **PagedAttention**: block-based memory allocation is omitted to keep cache operations in under 50 lines of PyTorch.
- **NVIDIA GPU/CUDA/Triton**: optimized for Apple Silicon and standard PyTorch.
- **Distributed serving / Tensor Parallelism / Pipeline Parallelism**: omitted for single-node simplicity.
- **Complex UI libraries / CSS frameworks**: uses simple vanilla HTML/CSS/JS.
