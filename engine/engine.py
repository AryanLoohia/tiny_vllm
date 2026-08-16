import time
import os
import resource
import traceback
import torch
from typing import List, Optional, Dict, Tuple
from engine.model import LLMModel
from engine.request import Request
from engine.scheduler import Scheduler
from engine.sampler import sample_token
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------------
# KV cache conversion helpers
# ---------------------------------------------------------------------------
def _extract_kv_cache(past_key_values) -> list:
    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        return [(past_key_values.key_cache[i], past_key_values.value_cache[i])
                for i in range(len(past_key_values.key_cache))]
    return [(layer[0], layer[1]) for layer in past_key_values]


def _to_dynamic_cache(kv_tuple_list) -> DynamicCache:
    legacy_format = tuple(kv_tuple_list)
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(legacy_format)
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(legacy_format):
        cache.update(k, v, layer_idx)
    return cache


def _clone_kv_cache(kv: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Deep-copy a request's KV cache (so prefix-cache sharing stays independent)."""
    return [(k.clone(), v.clone()) for (k, v) in kv]


def _truncate_kv(kv: List[Tuple[torch.Tensor, torch.Tensor]], length: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Truncate every layer's KV cache to `length` sequence positions (in place + return)."""
    for i, (k, v) in enumerate(kv):
        kv[i] = (k[:, :, :length, :].contiguous(), v[:, :, :length, :].contiguous())
    return kv


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class LLMEngine:
    """
    Core LLM inference engine.

    Features:
      - Continuous / static / sequential batching modes.
      - FIFO / SJF / priority scheduling.
      - Prefix KV caching for repeated prompts.
      - Chunked prefill (one chunk per step) to reduce peak memory + improve TTFT.
      - Live observability: per-request Gantt events + timeseries history
        (throughput, batch size, queue depth, memory).
    """

    DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

    def __init__(self, model_id: str = DEFAULT_MODEL, max_batch_size: int = 4):
        self.config = {
            "model_id": model_id,
            "max_batch_size": max_batch_size,
            "mode": "continuous",        # sequential | static | continuous
            "scheduler": "fifo",         # fifo | sjf | priority
            "precision": "fp32",         # fp32 | fp16 | int8
            "prefix_cache": True,
            "chunked_prefill": True,
            "chunk_size": 64,
            "speculative": False,        # speculative decoding on/off
            "spec_tokens": 4,             # K: draft tokens proposed per step
            "draft_model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            # main model defaults to the larger 360M so speculative decoding
            # (draft = 135M) actually produces a speedup.
        }
        self.model = LLMModel(model_id, precision=self.config["precision"])
        self.scheduler = Scheduler(max_batch_size=max_batch_size,
                                   policy=self.config["scheduler"],
                                   continuous=(self.config["mode"] == "continuous"))
        self.device = self.model.device

        # Draft model for speculative decoding (lazy-loaded)
        self.draft_model: Optional[LLMModel] = None

        # Observability
        self.step_count = 0
        self.total_batch_size = 0
        self.max_batch_size_seen = 0
        self.completed_requests: List[Request] = []
        self.history: List[dict] = []          # timeseries ring buffer
        self._history_cap = 240

        # Prefix cache: hash(prompt_tokens tuple) -> (kv_cache, last_logits)
        self.prefix_cache: Dict[tuple, Tuple[list, torch.Tensor]] = {}
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0

        self._last_step_time = time.time()
        self._last_throughput = 0.0

    # ------------------------------------------------------------------ config
    def apply_config(self, **changes) -> dict:
        """Apply runtime configuration changes. Returns the new config."""
        reload_model = False
        for k, v in changes.items():
            if k not in self.config:
                continue
            if k == "precision" and v != self.config["precision"]:
                reload_model = True
            if k == "model_id" and v != self.config["model_id"]:
                reload_model = True
            self.config[k] = v

        self.scheduler.max_batch_size = self.config["max_batch_size"]
        self.scheduler.policy = self.config["scheduler"]
        # mode mapping -> (max_batch, continuous)
        mode = self.config["mode"]
        if mode == "sequential":
            self.scheduler.max_batch_size = 1
            self.scheduler.continuous = True
        elif mode == "static":
            self.scheduler.max_batch_size = self.config["max_batch_size"]
            self.scheduler.continuous = False
        else:  # continuous
            self.scheduler.max_batch_size = self.config["max_batch_size"]
            self.scheduler.continuous = True

        if reload_model:
            try:
                self.model.reload(precision=self.config["precision"], model_id=self.config["model_id"])
                self.device = self.model.device
            except Exception as e:
                print(f"[ENGINE] model reload failed: {e}")
        # Lazily load / unload the draft model for speculative decoding
        if self.config["speculative"] and self.draft_model is None:
            try:
                self.draft_model = LLMModel(self.config["draft_model_id"], precision="fp16")
                print(f"[ENGINE] draft model loaded: {self.config['draft_model_id']}")
            except Exception as e:
                print(f"[ENGINE] draft model load failed: {e}")
                self.config["speculative"] = False
        elif not self.config["speculative"] and self.draft_model is not None:
            del self.draft_model
            self.draft_model = None
            print("[ENGINE] draft model unloaded")
        print(f"[ENGINE] config applied: {self.config}")
        return self.config

    # ------------------------------------------------------------------ request
    def add_request(self, prompt: str, max_tokens: int = 100, temperature: float = 1.0,
                    top_p: float = 1.0, top_k: int = 50, priority: int = 0,
                    request_id: Optional[str] = None) -> Request:
        if request_id is None:
            request_id = f"req_{int(time.time_ns())}"
        raw = self.model.tokenize(prompt, format_chat=True)
        prompt_tokens = raw
        if isinstance(prompt_tokens, dict) and "input_ids" in prompt_tokens:
            prompt_tokens = prompt_tokens["input_ids"]
        if hasattr(prompt_tokens, "tolist"):
            prompt_tokens = prompt_tokens.tolist()
        if len(prompt_tokens) > 0 and isinstance(prompt_tokens[0], list):
            prompt_tokens = prompt_tokens[0]

        request = Request(request_id=request_id, prompt=prompt, prompt_tokens=prompt_tokens,
                          max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                          top_k=top_k, priority=priority)
        self.scheduler.add_request(request)
        print(f"[ENGINE] + {request_id[:12]} queued | ptokens={len(prompt_tokens)} "
              f"| max={max_tokens} | prio={priority}")
        return request

    # ------------------------------------------------------------------ prefix cache (radix-style)
    def _find_longest_prefix(self, tokens: List[int]):
        """
        Radix-style shared-prefix lookup. Returns (matched_len, kv, logits) for the
        longest cached token-prefix that is a prefix of `tokens`, or (0, None, None).
        Unlike exact-match caching, this reuses KV across DIFFERENT prompts that
        share a common system-prompt prefix.
        """
        if not self.config["prefix_cache"] or len(tokens) == 0:
            return 0, None, None
        best_len, best_entry = 0, None
        for key, entry in self.prefix_cache.items():
            kl = len(key)
            if kl > best_len and kl <= len(tokens) and tuple(tokens[:kl]) == key:
                best_len, best_entry = kl, entry
        return best_len, best_entry[0], best_entry[1] if best_entry else None

    def _try_prefix_cache(self, req: Request) -> bool:
        """Attempt to serve (part of) prefill from the radix prefix cache.
        Returns True if prefill is FULLY satisfied (exact match)."""
        if not self.config["prefix_cache"] or len(req.prompt_tokens) == 0:
            return False
        matched_len, kv_cached, last_logits = self._find_longest_prefix(req.prompt_tokens)
        if matched_len == 0:
            self.prefix_cache_misses += 1
            return False

        req.start_time = req.start_time or time.time()

        # Reuse the matched prefix KV (truncated to matched_len, cloned).
        req.kv_cache = _truncate_kv(_clone_kv_cache(kv_cached), matched_len)
        if matched_len == len(req.prompt_tokens):
            # Exact match — no forward pass needed; sample from cached logits.
            req.prefix_cache_hit = True
            next_token = sample_token(last_logits, temperature=req.temperature,
                                      top_p=req.top_p, top_k=req.top_k)
            req.generated_tokens.append(next_token)
            req.prefill_end_time = time.time()
            req.ttft = req.prefill_end_time - req.arrival_time
            req.status = "DECODE"
            req.record_event("prefill_end", f"radix exact hit (len={matched_len}, 0 fwd)")
            req.record_event("decode", "first_token via cache")
            self.prefix_cache_hits += 1
            self._init_draft_cache(req)
            self._check_finish(req)
            print(f"[ENGINE] * RADIX EXACT HIT {req.request_id[:12]} | ttft={req.ttft:.3f}s")
            return True
        # Partial match — reuse prefix KV, prefill only the remaining suffix.
        req.prefix_cache_hit = True
        self.prefix_cache_hits += 1
        suffix = req.prompt_tokens[matched_len:]
        print(f"[ENGINE] * RADIX PARTIAL HIT {req.request_id[:12]} | reused={matched_len} suffix={len(suffix)}")
        # Partial entries may have logits=None; re-prefill the suffix to get fresh logits.
        self._prefill_suffix(req, suffix, matched_len)
        return True

    def _prefill_suffix(self, req: Request, suffix_tokens: List[int], start_pos: int):
        """Prefill a suffix of the prompt, appending to an already-populated KV cache."""
        input_ids = torch.tensor([suffix_tokens], dtype=torch.long, device=self.device)
        past_kv = _to_dynamic_cache(req.kv_cache)
        pos_ids = torch.tensor([[i for i in range(start_pos, start_pos + len(suffix_tokens))]],
                               dtype=torch.long, device=self.device)
        amask = torch.ones(1, start_pos + len(suffix_tokens), dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model.model(input_ids=input_ids, past_key_values=past_kv,
                                       attention_mask=amask, position_ids=pos_ids, use_cache=True)
            logits = outputs.logits[0, -1, :]
            req.kv_cache = _extract_kv_cache(outputs.past_key_values)
        self._finalize_prefill(req, logits)

    def _init_draft_cache(self, req: Request):
        """Prefill the draft model for a request (used by speculative decoding)."""
        if not self.config["speculative"] or self.draft_model is None:
            return
        input_ids = torch.tensor([req.prompt_tokens], dtype=torch.long, device=self.draft_model.device)
        try:
            with torch.no_grad():
                out = self.draft_model.model(input_ids, use_cache=True)
                req.draft_kv_cache = _extract_kv_cache(out.past_key_values)
        except Exception as e:
            print(f"[ENGINE] draft prefill failed: {e}")

    def _store_prefix_cache(self, req: Request, last_logits: torch.Tensor):
        if not self.config["prefix_cache"]:
            return
        key = tuple(req.prompt_tokens)
        if key not in self.prefix_cache:
            self.prefix_cache[key] = (_clone_kv_cache(req.kv_cache), last_logits.detach().clone())
        # Radix-style shared-prefix insertion: also cache the longest common prefix
        # shared with any existing entry, so future prompts that diverge can still
        # reuse the shared head (e.g. same system prompt, different questions).
        best_c = 0
        for k in list(self.prefix_cache.keys()):
            if k == key:
                continue
            c = 0
            m = min(len(k), len(key))
            while c < m and k[c] == key[c]:
                c += 1
            if c > best_c:
                best_c = c
        if best_c > 0:
            pkey = key[:best_c]
            if pkey not in self.prefix_cache and best_c < len(key):
                trunc = _truncate_kv(_clone_kv_cache(req.kv_cache), best_c)
                # logits at the end of the shared prefix = logits at position best_c-1.
                # We don't have those cached, so store a sentinel; partial hits always
                # re-prefill the suffix, so logits aren't needed for partial entries.
                self.prefix_cache[pkey] = (trunc, None)
        # bound the cache size (FIFO eviction)
        MAXCACHE = 64
        while len(self.prefix_cache) > MAXCACHE:
            self.prefix_cache.pop(next(iter(self.prefix_cache)))

    # ------------------------------------------------------------------ chunked prefill
    def _do_prefill(self, req: Request) -> bool:
        """
        Run prefill for a request. If chunked_prefill is enabled and the prompt
        is longer than chunk_size, processes ONE chunk per call and returns False
        until all chunks are done. Returns True when prefill is complete.
        """
        req.start_time = req.start_time or time.time()
        req.record_event("prefill_start")

        cs = self.config["chunk_size"]
        # Speculative decoding needs full single-shot prefill (no chunking) for the draft cache to align.
        use_chunks = (self.config["chunked_prefill"] and not self.config["speculative"]
                      and len(req.prompt_tokens) > cs)

        if not use_chunks:
            # Single-shot prefill (original behaviour)
            input_ids = torch.tensor([req.prompt_tokens], dtype=torch.long, device=self.device)
            with torch.no_grad():
                outputs = self.model.model(input_ids, use_cache=True)
                logits = outputs.logits[0, -1, :]
                req.kv_cache = _extract_kv_cache(outputs.past_key_values)
            self._finalize_prefill(req, logits)
            return True

        # Chunked prefill: process one chunk this step.
        start = req.prefill_chunks_done * cs
        end = min(start + cs, len(req.prompt_tokens))
        chunk = req.prompt_tokens[start:end]
        input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)

        with torch.no_grad():
            if req.kv_cache is None:
                outputs = self.model.model(input_ids, use_cache=True)
                req.kv_cache = _extract_kv_cache(outputs.past_key_values)
            else:
                past_kv = _to_dynamic_cache(req.kv_cache)
                seq_len = req.kv_cache[0][0].shape[2]
                pos_ids = torch.tensor([[i for i in range(seq_len, seq_len + len(chunk))]],
                                       dtype=torch.long, device=self.device)
                amask = torch.ones(1, seq_len + len(chunk), dtype=torch.long, device=self.device)
                outputs = self.model.model(input_ids=input_ids, past_key_values=past_kv,
                                           attention_mask=amask, position_ids=pos_ids, use_cache=True)
                req.kv_cache = _extract_kv_cache(outputs.past_key_values)

        req.prefill_chunks_done += 1
        if end >= len(req.prompt_tokens):
            # Final chunk — sample first token from last position's logits.
            with torch.no_grad():
                logits = outputs.logits[0, -1, :]
            self._finalize_prefill(req, logits)
            return True
        # Still chunking — stay in PREFILL phase.
        req.record_event("decode", f"chunk {req.prefill_chunks_done} ({len(chunk)} tok)")
        return False

    def _finalize_prefill(self, req: Request, last_logits: torch.Tensor):
        next_token = sample_token(last_logits, temperature=req.temperature,
                                  top_p=req.top_p, top_k=req.top_k)
        req.generated_tokens.append(next_token)
        req.prefill_end_time = time.time()
        req.ttft = req.prefill_end_time - req.arrival_time
        req.status = "DECODE"
        req.record_event("prefill_end", f"prompt_tokens={len(req.prompt_tokens)}")
        req.record_event("decode", "first_token")
        self._store_prefix_cache(req, last_logits)
        self._init_draft_cache(req)
        print(f"[ENGINE] # PREFILL {req.request_id[:12]} | ttft={req.ttft:.3f}s | "
              f"ptok={len(req.prompt_tokens)} | first={self.model.decode_token(next_token)!r}")
        self._check_finish(req)

    # ------------------------------------------------------------------ decode
    def _decode_single(self, req: Request):
        if self.config["speculative"] and self.draft_model is not None and req.draft_kv_cache is not None:
            self._decode_speculative(req)
            return
        # track batch-size observability even on the single path
        self.step_count += 1
        self.total_batch_size += 1
        self.max_batch_size_seen = max(self.max_batch_size_seen, 1)
        last_token_id = req.generated_tokens[-1]
        input_ids = torch.tensor([[last_token_id]], dtype=torch.long, device=self.device)
        past_kv = _to_dynamic_cache(req.kv_cache)
        seq_len = req.kv_cache[0][0].shape[2]
        position_ids = torch.tensor([[seq_len]], dtype=torch.long, device=self.device)
        attention_mask = torch.ones(1, seq_len + 1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model.model(input_ids=input_ids, past_key_values=past_kv,
                                       attention_mask=attention_mask, position_ids=position_ids,
                                       use_cache=True)
            logits = outputs.logits[0, -1, :]
            req.kv_cache = _extract_kv_cache(outputs.past_key_values)
        next_token = sample_token(logits, temperature=req.temperature, top_p=req.top_p, top_k=req.top_k)
        req.generated_tokens.append(next_token)
        req.record_event("decode", f"tok {len(req.generated_tokens)}")
        self._check_finish(req)

    # ------------------------------------------------------------------ speculative decode
    def _decode_speculative(self, req: Request):
        """
        Speculative decoding (single-request path):
          1. Draft model autoregressively proposes K candidate tokens.
          2. Main model verifies all K+1 positions in ONE forward pass.
          3. Accept the longest matching prefix; correct the first mismatch.
        Net effect: multiple tokens per main-model forward when the draft is accurate.
        """
        self.step_count += 1
        self.total_batch_size += 1
        self.max_batch_size_seen = max(self.max_batch_size_seen, 1)
        K = self.config["spec_tokens"]
        draft = self.draft_model
        ddev = draft.device
        main_dev = self.device

        # --- 1. Draft proposes K tokens (greedy) ---
        candidates = []
        last_tok = req.generated_tokens[-1]
        for _ in range(K):
            inp = torch.tensor([[last_tok]], dtype=torch.long, device=ddev)
            sl = req.draft_kv_cache[0][0].shape[2]
            pos = torch.tensor([[sl]], dtype=torch.long, device=ddev)
            am = torch.ones(1, sl + 1, dtype=torch.long, device=ddev)
            with torch.no_grad():
                out = draft.model(input_ids=inp, past_key_values=_to_dynamic_cache(req.draft_kv_cache),
                                  attention_mask=am, position_ids=pos, use_cache=True)
                req.draft_kv_cache = _extract_kv_cache(out.past_key_values)
                last_tok = int(torch.argmax(out.logits[0, -1, :]).item())
            candidates.append(last_tok)

        # --- 2. Main verifies [last_real_tok, c1, ..., cK] in one forward pass ---
        real_last = req.generated_tokens[-1]
        main_input = torch.tensor([[real_last] + candidates], dtype=torch.long, device=main_dev)
        msl = req.kv_cache[0][0].shape[2]
        pos_ids = torch.tensor([[i for i in range(msl, msl + len(candidates) + 1)]],
                               dtype=torch.long, device=main_dev)
        amask = torch.ones(1, msl + len(candidates) + 1, dtype=torch.long, device=main_dev)
        with torch.no_grad():
            out = self.model.model(input_ids=main_input, past_key_values=_to_dynamic_cache(req.kv_cache),
                                   attention_mask=amask, position_ids=pos_ids, use_cache=True)
            main_logits = out.logits[0]            # [K+1, vocab]
            new_kv = _extract_kv_cache(out.past_key_values)

        # --- 3. Verify & accept ---
        accepted = 0
        produced = []
        for i in range(K):
            predicted = int(torch.argmax(main_logits[i]).item())
            if predicted == candidates[i]:
                accepted += 1
                produced.append(predicted)
            else:
                produced.append(predicted)   # corrected token from main model
                break
        else:
            # all K accepted -> take bonus token from last position
            produced.append(int(torch.argmax(main_logits[K]).item()))

        # --- 4. Roll back KV caches to accepted length ---
        # Main KV grew by K+1; keep only len(produced).
        keep_main = msl + len(produced)
        req.kv_cache = _truncate_kv(new_kv, keep_main)
        # Draft KV grew by K; keep only `accepted` (draft tokens that matched).
        keep_draft = req.draft_kv_cache[0][0].shape[2] - (K - accepted)
        req.draft_kv_cache = _truncate_kv(req.draft_kv_cache, keep_draft)

        for t in produced:
            req.generated_tokens.append(t)
        req.record_event("decode", f"spec +{len(produced)} (acc {accepted}/{K})")
        print(f"[ENGINE] >> SPEC {req.request_id[:12]} | +{len(produced)} tok | "
              f"accepted {accepted}/{K} | {len(req.generated_tokens)}/{req.max_tokens}")
        self._check_finish(req)

    def _decode_batch(self, requests: List[Request]):
        from engine.kv_cache import pad_and_batch_kv_cache, slice_and_update_kv_cache
        batched_pkv, position_ids, attention_mask = pad_and_batch_kv_cache(requests, self.device)
        batched_pkv = _to_dynamic_cache(batched_pkv)
        last_tokens = [[r.generated_tokens[-1]] for r in requests]
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model.model(input_ids=input_ids, past_key_values=batched_pkv,
                                       attention_mask=attention_mask, position_ids=position_ids,
                                       use_cache=True)
            logits = outputs.logits[:, -1, :]
        slice_and_update_kv_cache(outputs.past_key_values, requests)
        for idx, req in enumerate(requests):
            next_token = sample_token(logits[idx], temperature=req.temperature,
                                      top_p=req.top_p, top_k=req.top_k)
            req.generated_tokens.append(next_token)
            req.record_event("decode", f"tok {len(req.generated_tokens)} (batch)")
            self._check_finish(req)

    def _check_finish(self, req: Request):
        if (req.generated_tokens and req.generated_tokens[-1] == self.model.get_eos_token_id()) \
                or len(req.generated_tokens) >= req.max_tokens:
            req.status = "FINISHED"
            req.finish_time = time.time()
            req.record_event("finish", f"tokens={len(req.generated_tokens)}")
            print(f"[ENGINE] v FINISHED {req.request_id[:12]} | tok={len(req.generated_tokens)} "
                  f"| lat={req.total_latency:.2f}s | {req.tokens_per_sec:.1f} t/s"
                  f"{' [cache]' if req.prefix_cache_hit else ''}")

    # ------------------------------------------------------------------ step
    def step(self) -> List[Request]:
        try:
            return self._step_inner()
        except Exception as e:
            print(f"[ENGINE] EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            return []

    def _step_inner(self) -> List[Request]:
        step_start = time.time()
        tokens_this_step = 0

        # 1. Admit new requests
        new_requests = self.scheduler.schedule()

        # 2. Prefill newly admitted requests (or restore from prefix cache)
        for req in new_requests:
            if self._try_prefix_cache(req):
                tokens_this_step += 1
                continue
            # Chunked prefill: do first chunk now; may stay PREFILL for later steps.
            self._do_prefill(req)
            tokens_this_step += 1

        # 3. Continue chunked-prefill requests that are still in PREFILL phase
        prefilling = [r for r in self.scheduler.active_requests if r.status == "PREFILL"]
        for req in prefilling:
            self._do_prefill(req)
            tokens_this_step += 1

        # 4. Decode step for DECODE-phase requests
        decode_reqs = [r for r in self.scheduler.active_requests
                       if r.status == "DECODE" and r.kv_cache is not None and r.generated_tokens]
        if decode_reqs:
            self.step_count += 1
            self.total_batch_size += len(decode_reqs)
            self.max_batch_size_seen = max(self.max_batch_size_seen, len(decode_reqs))
            before = sum(len(r.generated_tokens) for r in decode_reqs)
            if len(decode_reqs) == 1:
                self._decode_single(decode_reqs[0])
            else:
                self._decode_batch(decode_reqs)
            after = sum(len(r.generated_tokens) for r in decode_reqs)
            tokens_this_step += (after - before)

        # 5. Retire finished
        finished = self.scheduler.remove_finished()
        self.completed_requests.extend(finished)

        # 6. Observability: timeseries snapshot
        now = time.time()
        step_dur = max(now - self._last_step_time, 1e-6)
        self._last_throughput = tokens_this_step / step_dur
        self._last_step_time = now
        self._record_history(tokens_this_step, step_dur)
        return finished

    # ------------------------------------------------------------------ history
    def _record_history(self, tokens_this_step: int, step_dur: float):
        snap = {
            "t": round(time.time(), 3),
            "throughput": round(self._last_throughput, 2),
            "batch_size": len([r for r in self.scheduler.active_requests if r.status == "DECODE"]),
            "queue_depth": self.scheduler.get_queue_size(),
            "active": len(self.scheduler.active_requests),
            "memory_mb": round(self._mem_mb(), 1),
            "completed": len(self.completed_requests),
        }
        self.history.append(snap)
        if len(self.history) > self._history_cap:
            self.history = self.history[-self._history_cap:]

    def _mem_mb(self) -> float:
        # Resident set size of the process (MB)
        try:
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mb = rss_kb / 1024.0 if rss_kb < 10_000_000 else rss_kb / (1024.0 * 1024.0)
        except Exception:
            rss_mb = 0.0
        # MPS allocated memory (MB) if available
        try:
            if torch.backends.mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
                rss_mb = max(rss_mb, torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
        except Exception:
            pass
        return rss_mb

    # ------------------------------------------------------------------ status
    def get_metrics(self) -> dict:
        completed = self.completed_requests
        n = len(completed)
        def avg(seq):
            seq = [x for x in seq if x is not None and x > 0]
            return sum(seq) / len(seq) if seq else 0.0
        return {
            "num_completed": n,
            "avg_ttft": avg([r.ttft for r in completed]),
            "avg_latency": avg([r.total_latency for r in completed]),
            "avg_tokens_sec": avg([r.tokens_per_sec for r in completed]),
            "avg_queue_time": avg([r.queue_time for r in completed]),
            "avg_batch_size": self.total_batch_size / self.step_count if self.step_count > 0 else 0.0,
            "max_batch_size": self.max_batch_size_seen,
            "pending_queue_size": self.scheduler.get_queue_size(),
            "active_batch_size": len(self.scheduler.active_requests),
            "instant_throughput": round(self._last_throughput, 2),
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_misses": self.prefix_cache_misses,
            "prefix_cache_size": len(self.prefix_cache),
            "memory_mb": round(self._mem_mb(), 1),
            "precision": self.config["precision"],
            "mode": self.config["mode"],
            "scheduler": self.config["scheduler"],
            "max_batch_size_cfg": self.config["max_batch_size"],
            "chunked_prefill": self.config["chunked_prefill"],
            "chunk_size": self.config["chunk_size"],
            "kv_bytes_per_token": self.model.kv_bytes_per_token(),
            "speculative": self.config["speculative"],
            "spec_tokens": self.config["spec_tokens"],
            "draft_loaded": self.draft_model is not None,
            "prefix_cache_mode": "radix" if self.config["prefix_cache"] else "off",
        }

    def get_gantt(self) -> List[dict]:
        """Return per-request phase event logs for Gantt visualisation."""
        all_reqs = list(self.scheduler.active_requests) + list(self.completed_requests)
        # Keep most recent 20 for the chart
        recent = all_reqs[-20:]
        out = []
        for r in recent:
            out.append({
                "id": r.request_id,
                "prompt": r.prompt,
                "prefix_cache_hit": r.prefix_cache_hit,
                "status": r.status,
                "arrival_time": r.arrival_time,
                "start_time": r.start_time,
                "prefill_end_time": r.prefill_end_time,
                "finish_time": r.finish_time,
                "tokens": len(r.generated_tokens),
                "events": [(round(e[0], 3), e[1], e[2]) for e in r.events],
            })
        return out
