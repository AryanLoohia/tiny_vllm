from collections import deque
from engine.request import Request


class Scheduler:
    """Keeps a waiting queue and an active batch, and decides who runs next.

    Policies:
      - "fifo":     first in, first out (default)
      - "sjf":      shortest job first (fewest max_tokens first)
      - "priority": highest priority value first, FIFO as a tie-breaker

    Admission modes:
      - continuous=True:  refill a free slot the moment a request finishes
                          (this is what makes it continuous batching)
      - continuous=False: static batching; once a batch is running, nothing
                          new joins until the whole batch is done
    """

    def __init__(self, max_batch_size=4, policy="fifo", continuous=True):
        self.max_batch_size = max_batch_size
        self.policy = policy
        self.continuous = continuous
        self.waiting_queue = deque()
        self.active_requests = []

    def add_request(self, request):
        request.status = "WAITING"
        request.record_event("queue", f"prompt_tokens={len(request.prompt_tokens)}")
        self.waiting_queue.append(request)

    def get_queue_size(self):
        return len(self.waiting_queue)

    def _pick_next(self):
        """Pick the next request to run, following the current policy."""
        if self.policy == "sjf":
            idx = min(range(len(self.waiting_queue)),
                      key=lambda i: self.waiting_queue[i].max_tokens)
        elif self.policy == "priority":
            idx = max(range(len(self.waiting_queue)),
                      key=lambda i: self.waiting_queue[i].priority)
        else:
            idx = 0

        # Pop an arbitrary index out of a deque by rotating it to the front.
        dq = self.waiting_queue
        dq.rotate(-idx)
        req = dq.popleft()
        dq.rotate(idx)
        return req

    def schedule(self):
        """Promote waiting requests into free slots. Returns the newly promoted."""
        promoted = []

        # Static batching: don't add anyone until the current batch drains.
        if not self.continuous and len(self.active_requests) > 0:
            return promoted

        while len(self.active_requests) < self.max_batch_size and self.waiting_queue:
            req = self._pick_next()
            req.status = "PREFILL"
            self.active_requests.append(req)
            promoted.append(req)

        return promoted

    def remove_finished(self):
        """Pull finished requests out of the active batch (frees their slots)."""
        finished = [r for r in self.active_requests if r.status == "FINISHED"]
        self.active_requests = [r for r in self.active_requests if r.status != "FINISHED"]
        return finished


# Backwards-compatible alias.
FIFOScheduler = Scheduler
