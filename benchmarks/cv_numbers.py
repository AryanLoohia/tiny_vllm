import os, sys, time, statistics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.engine import LLMEngine

PROMPTS = [
    "Explain what a GPU is in one sentence.",
    "What is the capital of France and its famous landmarks?",
    "Write a short poem about coding in Python.",
    "Explain the concept of continuous batching in LLM serving.",
]
MAX_TOKENS = 30

def run_mode(mode, prompts, max_batch=4):
    eng = LLMEngine(max_batch_size=max_batch)
    eng.config["mode"] = mode
    eng.scheduler.max_batch_size = 1 if mode == "sequential" else max_batch
    eng.scheduler.continuous = (mode == "continuous")
    for p in prompts:
        eng.add_request(p, max_tokens=MAX_TOKENS)
    t0 = time.time()
    while eng.scheduler.active_requests or eng.scheduler.get_queue_size() > 0:
        eng.step()
    dur = time.time() - t0
    reqs = eng.completed_requests
    total = sum(len(r.generated_tokens) for r in reqs)
    ttft = statistics.mean(r.ttft for r in reqs if r.ttft)
    lat = statistics.mean(r.total_latency for r in reqs if r.total_latency)
    return dur, total / dur, ttft, lat

def concurrency_sweep():
    print("\n--- Concurrency Sweep (continuous) ---")
    base = PROMPTS
    for n in [1, 2, 4, 8]:
        prompts = (base * ((n // len(base)) + 1))[:n]
        eng = LLMEngine(max_batch_size=n)
        eng.config["mode"] = "continuous"
        eng.scheduler.max_batch_size = n
        eng.scheduler.continuous = True
        for p in prompts:
            eng.add_request(p, max_tokens=MAX_TOKENS)
        t0 = time.time()
        while eng.scheduler.active_requests or eng.scheduler.get_queue_size() > 0:
            eng.step()
        dur = time.time() - t0
        total = sum(len(r.generated_tokens) for r in eng.completed_requests)
        print(f"  concurrency={n}: dur={dur:.2f}s throughput={total/dur:.2f} t/s")

def radix_test():
    print("\n--- Radix Prefix Cache (cold vs hit) ---")
    eng = LLMEngine(max_batch_size=1)
    p = "Explain the concept of continuous batching in LLM serving systems."
    # cold
    eng.add_request(p, max_tokens=5)
    while eng.scheduler.active_requests or eng.scheduler.get_queue_size() > 0:
        eng.step()
    cold = eng.completed_requests[-1].ttft
    # hit (identical prompt)
    eng.add_request(p, max_tokens=5)
    while eng.scheduler.active_requests or eng.scheduler.get_queue_size() > 0:
        eng.step()
    hit = eng.completed_requests[-1].ttft
    # partial: shared prefix
    p2 = "Explain the concept of continuous batching in LLM serving systems and GPUs."
    eng.add_request(p2, max_tokens=5)
    while eng.scheduler.active_requests or eng.scheduler.get_queue_size() > 0:
        eng.step()
    part = eng.completed_requests[-1].ttft
    print(f"  cold TTFT={cold:.4f}s  hit TTFT={hit:.4f}s  partial-prefix TTFT={part:.4f}s  speedup={cold/hit:.1f}x")

def quant_test():
    print("\n--- Quantization precision sweep (continuous, 4 prompts) ---")
    for prec in ["fp32", "fp16", "int8"]:
        eng = LLMEngine(max_batch_size=4)
        eng.config["mode"] = "continuous"
        eng.config["precision"] = prec
        eng.scheduler.max_batch_size = 4
        eng.scheduler.continuous = True
        eng.model.reload(precision=prec)
        for p in PROMPTS:
            eng.add_request(p, max_tokens=20)
        t0 = time.time()
        while eng.scheduler.active_requests or eng.scheduler.get_queue_size() > 0:
            eng.step()
        dur = time.time() - t0
        total = sum(len(r.generated_tokens) for r in eng.completed_requests)
        print(f"  {prec}: dur={dur:.2f}s throughput={total/dur:.2f} t/s")

if __name__ == "__main__":
    print("=== Mode comparison (4 prompts x 30 tokens, SmolLM2-360M) ===")
    for mode in ["sequential", "static", "continuous"]:
        dur, tp, ttft, lat = run_mode(mode, PROMPTS)
        print(f"  {mode:12s} dur={dur:.2f}s throughput={tp:.2f} t/s avgTTFT={ttft:.3f}s avgLat={lat:.3f}s")
    concurrency_sweep()
    radix_test()
    quant_test()
