# Architecture: tiny-vLLM

This document details the system design, continuous batching mechanism, and KV cache management of the `tiny-vLLM` educational inference engine.

---

## 1. System Overview

`tiny-vLLM` is designed as a single-process server that handles concurrent LLM inference requests. It is structured into distinct layers to separate request reception, scheduling, batching, and tensor operations.

```
Client 
  ↓ (HTTP Request)
FastAPI Server (api.py)
  ↓ (Enqueue)
Request Queue (scheduler.py)
  ↓ (FIFO Schedule)
Scheduler (scheduler.py)
  ↓ (Schedule Active Batch)
Inference Engine (engine.py)
  ↓ (Padded & Concatenated)
KV Cache Manager (kv_cache.py)
  ↓ (Batched Tensors & Position IDs)
Hugging Face Causal LM (model.py)
```

---

## 2. Request Lifecycle

A request transitions through the following states:

1. **WAITING**: The request is created and appended to the scheduler's FIFO queue (`waiting_queue`).
2. **RUNNING**: The scheduler promotes the request to the `active_requests` list (when slot capacity is available).
   - **Prefill**: When first scheduled, a single forward pass processes the entire prompt text, initializes the request's specific KV cache, and generates the first token (TTFT).
3. **DECODING**: The request remains in the active batch. In each subsequent step, it is batched with other running requests to generate one token per iteration.
4. **FINISHED**: Generation terminates if the model produces the End-Of-Sequence (EOS) token or reaches the requested `max_tokens`. The request is removed from the active list.

---

## 3. Continuous Batching & KV Caching Internals

Continuous batching operates by scheduling new requests and removing finished ones dynamically at each decoding step, avoiding the waste of waiting for a static batch of requests to all finish.

### The KV Cache Batching Challenge
In a batch of $B$ decoding requests:
- Request 1 might have a prompt/history length of 12 tokens.
- Request 2 might have a prompt/history length of 5 tokens.

Because sequence lengths differ, their individual KV caches stored in the Request objects have different shapes along the sequence dimension:
- $K_1$: `[1, num_kv_heads, 12, head_dim]`
- $K_2$: `[1, num_kv_heads, 5, head_dim]`

To run them in a single batched PyTorch forward pass, we must stack them into a single tensor of shape `[2, num_kv_heads, 12, head_dim]`.

### The Solution: Left-Padding & Slicing
`tiny-vLLM` implements the following steps in `engine/kv_cache.py` during each decoding iteration:

1. **Left-Padding**: Find the maximum cache length $L_{max}$ in the batch. Prepend zero-tensors to the left of shorter KV caches:
   - For Request 2 (length 5), we prepend a zero-tensor of shape `[1, num_kv_heads, 7, head_dim]`, making its shape `[1, num_kv_heads, 12, head_dim]`.
2. **Concatenation**: Concatenate the padded tensors along the batch dimension (dimension 0) to form the model's `past_key_values`.
3. **Position IDs**: Because Rotary Position Embeddings (RoPE) compute rotations based on position indexes, we must supply explicit `position_ids` of shape `[B, 1]` indicating the true index of the next token for each request (i.e. `seq_len - 1`), overriding the default sequential index inferred from the padded cache size.
4. **Attention Mask**: Construct a mask of shape `[B, L_{max} + 1]`. For Request 2, the mask contains 7 zeros (ignoring the padded history) followed by 6 ones (5 past valid tokens + 1 current token).
5. **Model Run**: Run the forward pass with `past_key_values`, `position_ids`, and `attention_mask`.
6. **Unpadding**: The model appends the new keys/values, returning a batched KV cache of length $L_{max} + 1$. For each request, we slice out its index from the batch, remove the left padding (index range `pad_len:`), and save the updated cache of length `L_i + 1` back to the Request object.

---

## 4. Concurrency Model

FastAPI runs on an asynchronous event loop on a single thread. PyTorch model executions are CPU/MPS intensive and block execution for 10-100ms per step.
To prevent freezing the web server:
- The background generation loop runs `await asyncio.to_thread(engine.step)`.
- This delegates PyTorch forward passes to a thread pool worker, allowing FastAPI's main event loop to immediately handle incoming `/generate` or `/status` HTTP requests and stream tokens to clients without latency spikes.
