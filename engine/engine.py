import time
import resource
import traceback

import torch
from transformers.cache_utils import DynamicCache

from engine.model import LLMModel
from engine.request import Request
from engine.scheduler import Scheduler
from engine.sampler import sample_token


# ---------------------------------------------------------------------------
# Small helpers for working with the KV cache.
# The cache is just a list of (key, value) tensors, one pair per layer.
# HuggingFace sometimes gives us a DynamicCache object instead, so these
# helpers convert between the two shapes.
# ---------------------------------------------------------------------------

def _extract_kv_cache(past_key_values):
    """Pull the (key, value) list out of whatever HF returns."""
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        out = []
        for i in range(len(past_key_values.key_cache)):
            out.append((past_key_values.key_cache[i], past_key_values.value_cache[i]))
        return out
    return [(layer[0], layer[1]) for layer in past_key_values]


def _to_dynamic_cache(kv_list):
    """Turn our (key, value) list into a DynamicCache for the next forward pass."""
    legacy = tuple(kv_list)
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(legacy)
    cache = DynamicCache()
    for i, (k, v) in enumerate(legacy):
        cache.update(k, v, i)
    return cache


def _clone_kv_cache(kv):
    """Deep copy a KV cache so shared prefixes stay independent."""
    out = []
    for (k, v) in kv:
        out.append((k.clone(), v.clone()))
    return out


def _truncate_kv(kv, length):
    """Cut every layer's cache down to `length` sequence positions."""
    for i in range(len(kv)):
        k, v = kv[i]
        kv[i] = (k[:, :, :length, :].contiguous(), v[:, :, :length, :].contiguous())
    return kv


class LLMEngine:
    """
    The core inference engine.

    It runs a loop that:
      - admits requests from the scheduler
      - prefills them (maybe from a shared-prefix cache, maybe in chunks)
      - decodes tokens, one step at a time
      - retires finished requests

    All of continuous/static batching, prefix caching, chunked prefill and
    speculative decoding hang off this loop.
    """

    DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
    MAX_PREFIX_CACHE = 64
    HISTORY_CAP = 240

    def __init__(self, model_id=DEFAULT_MODEL, max_batch_size=4):
        self.config = {
            "model_id": model_id,
            "max_batch_size": max_batch_size,
            "mode": "continuous",        # sequential | static | continuous
            "scheduler": "fifo",         # fifo | sjf | priority
            "precision": "fp32",         # fp32 | fp16 | int8
            "prefix_cache": True,
            "chunked_prefill": True,
            "chunk_size": 64,
            "speculative": False,
            "spec_tokens": 4,            # K: draft tokens proposed per step
            "draft_model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        }

        self.model = LLMModel(model_id, precision=self.config["precision"])
        self.device = self.model.device
        self.scheduler = Scheduler(
            max_batch_size=max_batch_size,
            policy=self.config["scheduler"],
            continuous=(self.config["mode"] == "continuous"),
        )

        # Draft model is only loaded when speculative decoding is turned on.
        self.draft_model = None

        # Observability counters.
        self.step_count = 0
        self.total_batch_size = 0
        self.max_batch_size_seen = 0
        self.completed_requests = []
        self.history = []

        # Radix-style prefix cache: prompt token tuple -> (kv_cache, last_logits)
        self.prefix_cache = {}
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0

        self._last_step_time = time.time()
        self._last_throughput = 0.0

    # ------------------------------------------------------------------
    # Config changes
    # ------------------------------------------------------------------
    def apply_config(self, **changes):
        """Apply runtime config changes and return the new config."""
        reload_model = False

        for key, value in changes.items():
            if key not in self.config:
                continue
            if key == "precision" and value != self.config["precision"]:
                reload_model = True
            if key == "model_id" and value != self.config["model_id"]:
                reload_model = True
            self.config[key] = value

        self.scheduler.max_batch_size = self.config["max_batch_size"]
        self.scheduler.policy = self.config["scheduler"]

        mode = self.config["mode"]
        if mode == "sequential":
            self.scheduler.max_batch_size = 1
            self.scheduler.continuous = True
        elif mode == "static":
            self.scheduler.continuous = False
        else:
            self.scheduler.continuous = True

        if reload_model:
            try:
                self.model.reload(precision=self.config["precision"],
                                  model_id=self.config["model_id"])
                self.device = self.model.device
            except Exception as e:
                print(f"[ENGINE] model reload failed: {e}")

        self._sync_draft_model()
        print(f"[ENGINE] config applied: {self.config}")
        return self.config

    def _sync_draft_model(self):
        """Load or unload the draft model depending on the speculative flag."""
        if not self.config["speculative"]:
            if self.draft_model is not None:
                del self.draft_model
                self.draft_model = None
                print("[ENGINE] draft model unloaded")
            return

        if self.draft_model is not None:
            return

        try:
            self.draft_model = LLMModel(self.config["draft_model_id"], precision="fp16")
            print(f"[ENGINE] draft model loaded: {self.config['draft_model_id']}")
        except Exception as e:
            print(f"[ENGINE] draft model load failed: {e}")
            self.config["speculative"] = False

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------
    def add_request(self, prompt, max_tokens=100, temperature=1.0,
                    top_p=1.0, top_k=50, priority=0, request_id=None):
        if request_id is None:
            request_id = f"req_{int(time.time_ns())}"

        prompt_tokens = self.model.tokenize(prompt, format_chat=True)
        if isinstance(prompt_tokens, dict) and "input_ids" in prompt_tokens:
            prompt_tokens = prompt_tokens["input_ids"]
        if hasattr(prompt_tokens, "tolist"):
            prompt_tokens = prompt_tokens.tolist()
        if len(prompt_tokens) > 0 and isinstance(prompt_tokens[0], list):
            prompt_tokens = prompt_tokens[0]

        request = Request(
            request_id=request_id, prompt=prompt, prompt_tokens=prompt_tokens,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, priority=priority,
        )
        self.scheduler.add_request(request)
        print(f"[ENGINE] + {request_id[:12]} queued | ptokens={len(prompt_tokens)} "
              f"| max={max_tokens} | prio={priority}")
        return request

    # ------------------------------------------------------------------
    # Radix-style shared-prefix cache
    # ------------------------------------------------------------------
    def _find_longest_prefix(self, tokens):
        """Return (matched_len, kv, logits) for the longest cached prefix of tokens."""
        if not self.config["prefix_cache"] or len(tokens) == 0:
            return 0, None, None

        best_len = 0
        best_entry = None
        for key, entry in self.prefix_cache.items():
            kl = len(key)
            too_long = kl > len(tokens)
            not_better = kl <= best_len
            if too_long or not_better:
                continue
            if tuple(tokens[:kl]) == key:
                best_len = kl
                best_entry = entry

        if best_entry is None:
            return 0, None, None
        return best_len, best_entry[0], best_entry[1]

    def _try_prefix_cache(self, req):
        """Serve prefill from the prefix cache if possible.

        Returns True if the request is fully prefilled (exact or partial hit),
        False if nothing was reused and a normal prefill is still needed.
        """
        if not self.config["prefix_cache"] or len(req.prompt_tokens) == 0:
            return False

        matched_len, kv_cached, last_logits = self._find_longest_prefix(req.prompt_tokens)
        if matched_len == 0:
            self.prefix_cache_misses += 1
            return False

        req.start_time = req.start_time or time.time()
        req.kv_cache = _truncate_kv(_clone_kv_cache(kv_cached), matched_len)

        # Exact match: no forward pass needed, sample straight from cached logits.
        if matched_len == len(req.prompt_tokens):
            req.prefix_cache_hit = True
            self.prefix_cache_hits += 1
            next_token = sample_token(last_logits, temperature=req.temperature,
                                      top_p=req.top_p, top_k=req.top_k)
            req.generated_tokens.append(next_token)
            req.prefill_end_time = time.time()
            req.ttft = req.prefill_end_time - req.arrival_time
            req.status = "DECODE"
            req.record_event("prefill_end", f"radix exact hit (len={matched_len}, 0 fwd)")
            req.record_event("decode", "first_token via cache")
            self._init_draft_cache(req)
            self._check_finish(req)
            print(f"[ENGINE] * RADIX EXACT HIT {req.request_id[:12]} | ttft={req.ttft:.3f}s")
            return True

        # Partial match: reuse the prefix KV, prefill only the diverging suffix.
        req.prefix_cache_hit = True
        self.prefix_cache_hits += 1
        suffix = req.prompt_tokens[matched_len:]
        print(f"[ENGINE] * RADIX PARTIAL HIT {req.request_id[:12]} "
              f"| reused={matched_len} suffix={len(suffix)}")
        self._prefill_suffix(req, suffix, matched_len)
        return True

    def _prefill_suffix(self, req, suffix_tokens, start_pos):
        """Prefill just the suffix of a prompt onto an existing KV cache."""
        input_ids = torch.tensor([suffix_tokens], dtype=torch.long, device=self.device)
        past_kv = _to_dynamic_cache(req.kv_cache)
        pos_ids = torch.tensor(
            [[i for i in range(start_pos, start_pos + len(suffix_tokens))]],
            dtype=torch.long, device=self.device,
        )
        amask = torch.ones(1, start_pos + len(suffix_tokens), dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids, past_key_values=past_kv,
                attention_mask=amask, position_ids=pos_ids, use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            req.kv_cache = _extract_kv_cache(outputs.past_key_values)

        self._finalize_prefill(req, logits)

    def _init_draft_cache(self, req):
        """Prefill the draft model so speculative decoding has a starting cache."""
        if not self.config["speculative"] or self.draft_model is None:
            return
        input_ids = torch.tensor([req.prompt_tokens], dtype=torch.long,
                                 device=self.draft_model.device)
        try:
            with torch.no_grad():
                out = self.draft_model.model(input_ids, use_cache=True)
                req.draft_kv_cache = _extract_kv_cache(out.past_key_values)
        except Exception as e:
            print(f"[ENGINE] draft prefill failed: {e}")

    def _store_prefix_cache(self, req, last_logits):
        """Save this request's KV cache so future prompts can reuse the prefix."""
        if not self.config["prefix_cache"]:
            return

        key = tuple(req.prompt_tokens)
        if key not in self.prefix_cache:
            self.prefix_cache[key] = (_clone_kv_cache(req.kv_cache), last_logits.detach().clone())

        # Also cache the longest prefix shared with any other entry, so prompts
        # that diverge mid-way can still reuse the shared head.
        best_common = 0
        for other_key in list(self.prefix_cache.keys()):
            if other_key == key:
                continue
            common = self._common_prefix_len(other_key, key)
            if common > best_common:
                best_common = common

        if best_common > 0 and best_common < len(key):
            pkey = key[:best_common]
            if pkey not in self.prefix_cache:
                trunc = _truncate_kv(_clone_kv_cache(req.kv_cache), best_common)
                # We don't have the logits at the split point, so store None.
                # Partial hits always re-prefill the suffix, so that's fine.
                self.prefix_cache[pkey] = (trunc, None)

        # Keep the cache bounded (FIFO eviction).
        while len(self.prefix_cache) > self.MAX_PREFIX_CACHE:
            self.prefix_cache.pop(next(iter(self.prefix_cache)))

    @staticmethod
    def _common_prefix_len(a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    # ------------------------------------------------------------------
    # Prefill (plain or chunked)
    # ------------------------------------------------------------------
    def _do_prefill(self, req):
        """Prefill a request.

        With chunked prefill on, this processes ONE chunk per call and returns
        False until every chunk is done. Returns True when prefill is complete.
        """
        req.start_time = req.start_time or time.time()
        req.record_event("prefill_start")

        chunk_size = self.config["chunk_size"]
        long_enough = len(req.prompt_tokens) > chunk_size
        use_chunks = (self.config["chunked_prefill"]
                      and not self.config["speculative"]
                      and long_enough)

        if not use_chunks:
            return self._prefill_single_shot(req)

        return self._prefill_next_chunk(req, chunk_size)

    def _prefill_single_shot(self, req):
        input_ids = torch.tensor([req.prompt_tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model.model(input_ids, use_cache=True)
            logits = outputs.logits[0, -1, :]
            req.kv_cache = _extract_kv_cache(outputs.past_key_values)
        self._finalize_prefill(req, logits)
        return True

    def _prefill_next_chunk(self, req, chunk_size):
        start = req.prefill_chunks_done * chunk_size
        end = min(start + chunk_size, len(req.prompt_tokens))
        chunk = req.prompt_tokens[start:end]
        input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)

        with torch.no_grad():
            if req.kv_cache is None:
                outputs = self.model.model(input_ids, use_cache=True)
                req.kv_cache = _extract_kv_cache(outputs.past_key_values)
            else:
                outputs = self._continue_prefill(req, chunk, input_ids)

        req.prefill_chunks_done += 1

        if end >= len(req.prompt_tokens):
            # Last chunk: grab the logits and finish prefill.
            with torch.no_grad():
                logits = outputs.logits[0, -1, :]
            self._finalize_prefill(req, logits)
            return True

        # Still chunking; stay in PREFILL for the next engine step.
        req.record_event("decode", f"chunk {req.prefill_chunks_done} ({len(chunk)} tok)")
        return False

    def _continue_prefill(self, req, chunk, input_ids):
        past_kv = _to_dynamic_cache(req.kv_cache)
        seq_len = req.kv_cache[0][0].shape[2]
        pos_ids = torch.tensor(
            [[i for i in range(seq_len, seq_len + len(chunk))]],
            dtype=torch.long, device=self.device,
        )
        amask = torch.ones(1, seq_len + len(chunk), dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids, past_key_values=past_kv,
                attention_mask=amask, position_ids=pos_ids, use_cache=True,
            )
            req.kv_cache = _extract_kv_cache(outputs.past_key_values)
        return outputs

    def _finalize_prefill(self, req, last_logits):
        """Sample the first token, mark the request as decoding, store cache."""
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

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    def _decode_single(self, req):
        if self._should_decode_speculatively(req):
            self._decode_speculative(req)
            return

        self.step_count += 1
        self.total_batch_size += 1
        self.max_batch_size_seen = max(self.max_batch_size_seen, 1)

        last_token_id = req.generated_tokens[-1]
        input_ids = torch.tensor([[last_token_id]], dtype=torch.long, device=self.device)
        past_kv = _to_dynamic_cache(req.kv_cache)
        seq_len = req.kv_cache[0][0].shape[2]
        position_id = torch.tensor([[seq_len]], dtype=torch.long, device=self.device)
        attention_mask = torch.ones(1, seq_len + 1, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids, past_key_values=past_kv,
                attention_mask=attention_mask, position_ids=position_id, use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            req.kv_cache = _extract_kv_cache(outputs.past_key_values)

        next_token = sample_token(logits, temperature=req.temperature,
                                  top_p=req.top_p, top_k=req.top_k)
        req.generated_tokens.append(next_token)
        req.record_event("decode", f"tok {len(req.generated_tokens)}")
        self._check_finish(req)

    def _should_decode_speculatively(self, req):
        if not self.config["speculative"]:
            return False
        if self.draft_model is None:
            return False
        if req.draft_kv_cache is None:
            return False
        return True

    def _decode_speculative(self, req):
        """Speculative decoding for a single request.

        1. The draft model proposes K tokens autoregressively.
        2. The main model verifies all K+1 positions in one forward pass.
        3. We accept the longest matching prefix and correct the first mismatch.
        """
        self.step_count += 1
        self.total_batch_size += 1
        self.max_batch_size_seen = max(self.max_batch_size_seen, 1)

        K = self.config["spec_tokens"]
        draft = self.draft_model
        draft_device = draft.device
        main_device = self.device

        # 1. Draft proposes K tokens (greedy).
        candidates = []
        last_tok = req.generated_tokens[-1]
        for _ in range(K):
            inp = torch.tensor([[last_tok]], dtype=torch.long, device=draft_device)
            sl = req.draft_kv_cache[0][0].shape[2]
            pos = torch.tensor([[sl]], dtype=torch.long, device=draft_device)
            am = torch.ones(1, sl + 1, dtype=torch.long, device=draft_device)
            with torch.no_grad():
                out = draft.model(
                    input_ids=inp,
                    past_key_values=_to_dynamic_cache(req.draft_kv_cache),
                    attention_mask=am, position_ids=pos, use_cache=True,
                )
                req.draft_kv_cache = _extract_kv_cache(out.past_key_values)
                last_tok = int(torch.argmax(out.logits[0, -1, :]).item())
            candidates.append(last_tok)

        # 2. Main verifies [real_last, c1, ..., cK] in one forward pass.
        real_last = req.generated_tokens[-1]
        main_input = torch.tensor([[real_last] + candidates], dtype=torch.long, device=main_device)
        msl = req.kv_cache[0][0].shape[2]
        pos_ids = torch.tensor(
            [[i for i in range(msl, msl + len(candidates) + 1)]],
            dtype=torch.long, device=main_device,
        )
        amask = torch.ones(1, msl + len(candidates) + 1, dtype=torch.long, device=main_device)
        with torch.no_grad():
            out = self.model.model(
                input_ids=main_input,
                past_key_values=_to_dynamic_cache(req.kv_cache),
                attention_mask=amask, position_ids=pos_ids, use_cache=True,
            )
            main_logits = out.logits[0]      # [K+1, vocab]
            new_kv = _extract_kv_cache(out.past_key_values)

        # 3. Verify and accept the longest matching prefix.
        produced, accepted = self._verify_candidates(main_logits, candidates, K)

        # 4. Roll back both KV caches to the accepted length.
        keep_main = msl + len(produced)
        req.kv_cache = _truncate_kv(new_kv, keep_main)
        keep_draft = req.draft_kv_cache[0][0].shape[2] - (K - accepted)
        req.draft_kv_cache = _truncate_kv(req.draft_kv_cache, keep_draft)

        for t in produced:
            req.generated_tokens.append(t)
        req.record_event("decode", f"spec +{len(produced)} (acc {accepted}/{K})")
        print(f"[ENGINE] >> SPEC {req.request_id[:12]} | +{len(produced)} tok | "
              f"accepted {accepted}/{K} | {len(req.generated_tokens)}/{req.max_tokens}")
        self._check_finish(req)

    def _verify_candidates(self, main_logits, candidates, K):
        """Compare draft candidates against the main model's predictions."""
        produced = []
        accepted = 0
        all_accepted = True

        for i in range(K):
            predicted = int(torch.argmax(main_logits[i]).item())
            if predicted == candidates[i]:
                accepted += 1
                produced.append(predicted)
            else:
                produced.append(predicted)   # corrected by the main model
                all_accepted = False
                break

        # If every draft token was accepted, take the bonus token from position K.
        if all_accepted:
            produced.append(int(torch.argmax(main_logits[K]).item()))

        return produced, accepted

    def _decode_batch(self, requests):
        from engine.kv_cache import pad_and_batch_kv_cache, slice_and_update_kv_cache

        batched_pkv, position_ids, attention_mask = pad_and_batch_kv_cache(requests, self.device)
        batched_pkv = _to_dynamic_cache(batched_pkv)

        last_tokens = [[r.generated_tokens[-1]] for r in requests]
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids, past_key_values=batched_pkv,
                attention_mask=attention_mask, position_ids=position_ids, use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
        slice_and_update_kv_cache(outputs.past_key_values, requests)

        for idx, req in enumerate(requests):
            next_token = sample_token(logits[idx], temperature=req.temperature,
                                      top_p=req.top_p, top_k=req.top_k)
            req.generated_tokens.append(next_token)
            req.record_event("decode", f"tok {len(req.generated_tokens)} (batch)")
            self._check_finish(req)

    def _check_finish(self, req):
        """Mark a request finished if it hit EOS or its token limit."""
        hit_eos = (req.generated_tokens and
                   req.generated_tokens[-1] == self.model.get_eos_token_id())
        hit_limit = len(req.generated_tokens) >= req.max_tokens
        if not (hit_eos or hit_limit):
            return

        req.status = "FINISHED"
        req.finish_time = time.time()
        req.record_event("finish", f"tokens={len(req.generated_tokens)}")
        tag = " [cache]" if req.prefix_cache_hit else ""
        print(f"[ENGINE] v FINISHED {req.request_id[:12]} | tok={len(req.generated_tokens)} "
              f"| lat={req.total_latency:.2f}s | {req.tokens_per_sec:.1f} t/s{tag}")

    # ------------------------------------------------------------------
    # The main step
    # ------------------------------------------------------------------
    def step(self):
        try:
            return self._step_inner()
        except Exception as e:
            print(f"[ENGINE] EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
            return []

    def _step_inner(self):
        step_start = time.time()
        tokens_this_step = 0

        # 1. Admit new requests from the scheduler.
        new_requests = self.scheduler.schedule()

        # 2. Prefill newly admitted requests (maybe served from the prefix cache).
        for req in new_requests:
            if self._try_prefix_cache(req):
                tokens_this_step += 1
                continue
            self._do_prefill(req)
            tokens_this_step += 1

        # 3. Keep chunking any requests still in the PREFILL phase.
        prefilling = [r for r in self.scheduler.active_requests if r.status == "PREFILL"]
        for req in prefilling:
            self._do_prefill(req)
            tokens_this_step += 1

        # 4. Decode one step for everything in the DECODE phase.
        decode_reqs = [r for r in self.scheduler.active_requests if self._is_decodable(r)]
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

        # 5. Retire finished requests.
        finished = self.scheduler.remove_finished()
        self.completed_requests.extend(finished)

        # 6. Record a timeseries snapshot.
        now = time.time()
        step_dur = max(now - self._last_step_time, 1e-6)
        self._last_throughput = tokens_this_step / step_dur
        self._last_step_time = now
        self._record_history(tokens_this_step, step_dur)
        return finished

    @staticmethod
    def _is_decodable(req):
        if req.status != "DECODE":
            return False
        if req.kv_cache is None:
            return False
        if not req.generated_tokens:
            return False
        return True

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def _record_history(self, tokens_this_step, step_dur):
        decoding = [r for r in self.scheduler.active_requests if r.status == "DECODE"]
        snap = {
            "t": round(time.time(), 3),
            "throughput": round(self._last_throughput, 2),
            "batch_size": len(decoding),
            "queue_depth": self.scheduler.get_queue_size(),
            "active": len(self.scheduler.active_requests),
            "memory_mb": round(self._mem_mb(), 1),
            "completed": len(self.completed_requests),
        }
        self.history.append(snap)
        if len(self.history) > self.HISTORY_CAP:
            self.history = self.history[-self.HISTORY_CAP:]

    def _mem_mb(self):
        """Resident memory of the process, plus MPS memory if available."""
        try:
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if rss_kb < 10_000_000:
                rss_mb = rss_kb / 1024.0
            else:
                rss_mb = rss_kb / (1024.0 * 1024.0)
        except Exception:
            rss_mb = 0.0

        try:
            if torch.backends.mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
                rss_mb = max(rss_mb, torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
        except Exception:
            pass
        return rss_mb

    # ------------------------------------------------------------------
    # Status / metrics
    # ------------------------------------------------------------------
    def get_metrics(self):
        completed = self.completed_requests
        n = len(completed)

        def avg(seq):
            good = [x for x in seq if x is not None and x > 0]
            if not good:
                return 0.0
            return sum(good) / len(good)

        avg_batch = self.total_batch_size / self.step_count if self.step_count > 0 else 0.0

        return {
            "num_completed": n,
            "avg_ttft": avg([r.ttft for r in completed]),
            "avg_latency": avg([r.total_latency for r in completed]),
            "avg_tokens_sec": avg([r.tokens_per_sec for r in completed]),
            "avg_queue_time": avg([r.queue_time for r in completed]),
            "avg_batch_size": avg_batch,
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

    def get_gantt(self):
        """Per-request phase event logs for the Gantt chart."""
        all_reqs = list(self.scheduler.active_requests) + list(self.completed_requests)
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
