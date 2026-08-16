from dataclasses import dataclass, field
import time
from typing import List, Tuple, Optional, Any
import torch

@dataclass
class Request:
    request_id: str
    prompt: str
    prompt_tokens: List[int]
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    priority: int = 0

    # State tracking
    status: str = "WAITING"        # WAITING, PREFILL, DECODE, FINISHED
    generated_tokens: List[int] = field(default_factory=list)

    # Timing and performance metrics
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    ttft: Optional[float] = None  # Time-to-First-Token
    finish_time: Optional[float] = None

    # Request-specific KV cache: list of (key, value) tuples per transformer layer
    kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    # Draft-model KV cache (used only when speculative decoding is enabled)
    draft_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    # --- Feature support fields ---
    prefix_cache_hit: bool = False
    prefill_chunks_done: int = 0   # for chunked prefill

    # Gantt / phase event log: list of (timestamp, event_type, detail)
    # event_type in {"prefill_start","prefill_end","decode","finish","queue"}
    events: List[Tuple[float, str, str]] = field(default_factory=list)

    def record_event(self, etype: str, detail: str = ""):
        self.events.append((time.time(), etype, detail))

    @property
    def total_latency(self) -> Optional[float]:
        if self.finish_time is not None:
            return self.finish_time - self.arrival_time
        return None

    @property
    def queue_time(self) -> Optional[float]:
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return None

    @property
    def tokens_per_sec(self) -> float:
        if self.finish_time is not None and self.start_time is not None:
            duration = self.finish_time - self.start_time
            if duration > 0 and len(self.generated_tokens) > 0:
                return len(self.generated_tokens) / duration
        return 0.0

    @property
    def seq_len(self) -> int:
        prompt_done = len(self.prompt_tokens) if self.kv_cache is not None else 0
        return prompt_done + len(self.generated_tokens)

    @property
    def prompt_kv_len(self) -> int:
        """Number of prompt tokens currently materialized in the KV cache."""
        if self.kv_cache is None or len(self.kv_cache) == 0:
            return 0
        return self.kv_cache[0][0].shape[2]
