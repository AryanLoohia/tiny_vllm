import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.engine import LLMEngine

def main():
    eng = LLMEngine(max_batch_size=4)
    # continuous, small gen
    for p in ["What is a GPU in one sentence?", "Name the capital of France."]:
        eng.add_request(p, max_tokens=8, temperature=0.0)
    t0 = time.time()
    while eng.scheduler.active_requests or eng.scheduler.get_queue_size() > 0:
        eng.step()
    print("continuous done in %.2fs, completed=%d" % (time.time()-t0, len(eng.completed_requests)))
    # radix cache hit (repeat same prompt)
    eng.add_request("What is a GPU in one sentence?", max_tokens=4, temperature=0.0)
    while eng.scheduler.active_requests or eng.scheduler.get_queue_size() > 0:
        eng.step()
    r = eng.completed_requests[-1]
    print("cache hit:", r.prefix_cache_hit, "ttft=%.4f" % (r.ttft or 0))
    # speculative decode path
    eng.apply_config(speculative=True, spec_tokens=3, mode="sequential")
    eng.add_request("Write a short poem about the sea.", max_tokens=12, temperature=0.0)
    while eng.scheduler.active_requests or eng.scheduler.get_queue_size() > 0:
        eng.step()
    r = eng.completed_requests[-1]
    print("spec tokens:", len(r.generated_tokens))
    print("ALL OK")

if __name__ == "__main__":
    main()
