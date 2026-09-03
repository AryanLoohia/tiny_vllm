from dataclasses import dataclass, field
import time
import torch


@dataclass
class Request:
    """One generation request and all the state that goes with it."""

    request_id: str
    prompt: str
    prompt_tokens: list
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    priority: int = 0

    # Lifecycle: WAITING -> PREFILL -> DECODE -> FINISHED
    status: str = "WAITING"
    generated_tokens: list = field(default_factory=list)

    # Timing.
    arrival_time: float = field(default_factory=time.time)
    start_time: float = None
    prefill_end_time: float = None
    ttft: float = None              # time to first token
    finish_time: float = None

    # KV cache for this request: list of (key, value) tensors, one per layer.
    kv_cache: list = None
    # KV cache for the draft model (only used by speculative decoding).
    draft_kv_cache: list = None

    # Feature flags / progress.
    prefix_cache_hit: bool = False
    prefill_chunks_done: int = 0

    # Gantt event log: list of (timestamp, event_type, detail).
    events: list = field(default_factory=list)

    def record_event(self, etype, detail=""):
        self.events.append((time.time(), etype, detail))

    @property
    def total_latency(self):
        if self.finish_time is not None:
            return self.finish_time - self.arrival_time
        return None

    @property
    def queue_time(self):
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return None

    @property
    def tokens_per_sec(self):
        if self.finish_time is None or self.start_time is None:
            return 0.0
        duration = self.finish_time - self.start_time
        if duration <= 0 or len(self.generated_tokens) == 0:
            return 0.0
        return len(self.generated_tokens) / duration

    @property
    def seq_len(self):
        prompt_done = len(self.prompt_tokens) if self.kv_cache is not None else 0
        return prompt_done + len(self.generated_tokens)

    @property
    def prompt_kv_len(self):
        """How many prompt tokens are currently in the KV cache."""
        if self.kv_cache is None or len(self.kv_cache) == 0:
            return 0
        return self.kv_cache[0][0].shape[2]
