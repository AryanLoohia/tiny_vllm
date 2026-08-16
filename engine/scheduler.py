from collections import deque
from typing import List
from engine.request import Request

class Scheduler:
    """
    Manages waiting and active requests.

    Policies:
      - "fifo":     First-In-First-Out (default).
      - "sjf":      Shortest-Job-First — promotes the request with the fewest
                    max_tokens first (minimises average waiting time).
      - "priority": Promotes highest `priority` value first (stable tie-break FIFO).

    Admission modes (controlled by `continuous`):
      - continuous=True:  New requests are admitted into free slots as soon as
                          any active request finishes (true continuous batching).
      - continuous=False: Static batching — once a batch is promoted, no new
                          requests enter until the ENTIRE active batch is done.
    """
    def __init__(self, max_batch_size: int = 4, policy: str = "fifo", continuous: bool = True):
        self.max_batch_size = max_batch_size
        self.policy = policy
        self.continuous = continuous
        self.waiting_queue: deque[Request] = deque()
        self.active_requests: List[Request] = []

    def add_request(self, request: Request) -> None:
        request.status = "WAITING"
        request.record_event("queue", f"prompt_tokens={len(request.prompt_tokens)}")
        self.waiting_queue.append(request)

    def get_queue_size(self) -> int:
        return len(self.waiting_queue)

    def _pick_next(self) -> Request:
        """Select the next request from the waiting queue according to policy."""
        if self.policy == "sjf":
            idx = min(range(len(self.waiting_queue)),
                      key=lambda i: self.waiting_queue[i].max_tokens)
        elif self.policy == "priority":
            idx = max(range(len(self.waiting_queue)),
                      key=lambda i: self.waiting_queue[i].priority)
        else:  # fifo
            idx = 0
        # pop arbitrary index from deque
        dq = self.waiting_queue
        dq.rotate(-idx)
        req = dq.popleft()
        dq.rotate(idx)
        return req

    def schedule(self) -> List[Request]:
        newly_promoted = []
        # Static batching: do not admit new requests until the current batch drains.
        if not self.continuous and len(self.active_requests) > 0:
            return newly_promoted
        while len(self.active_requests) < self.max_batch_size and self.waiting_queue:
            request = self._pick_next()
            request.status = "PREFILL"
            self.active_requests.append(request)
            newly_promoted.append(request)
        return newly_promoted

    def remove_finished(self) -> List[Request]:
        finished = [r for r in self.active_requests if r.status == "FINISHED"]
        self.active_requests = [r for r in self.active_requests if r.status != "FINISHED"]
        return finished

# Backwards-compatible alias
FIFOScheduler = Scheduler
