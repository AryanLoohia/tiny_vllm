import os
import sys
import time
from typing import List

# Add the workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.engine import LLMEngine
from engine.request import Request

TEST_PROMPTS = [
    "Explain what a GPU is in one sentence.",
    "What is the capital of France and what are its famous landmarks?",
    "Write a short poem about coding in Python.",
    "Explain the concept of continuous batching in LLM serving.",
    "Compare and contrast CPU and GPU architectures for AI.",
    "What is deep learning and how does it relate to machine learning?"
]

MAX_TOKENS = 30  # Keep generations small for benchmarking on Apple Silicon

def run_sequential(prompts: List[str]) -> List[Request]:
    print("\n--- Running Sequential Benchmark ---")
    completed_requests = []
    
    start_time = time.time()
    for idx, prompt in enumerate(prompts):
        # Fresh engine per prompt to reset state/KV caches cleanly
        engine = LLMEngine(max_batch_size=1)
        req = engine.add_request(prompt, max_tokens=MAX_TOKENS)
        
        print(f"Processing prompt {idx + 1}/{len(prompts)} sequentially...")
        while len(engine.scheduler.active_requests) > 0 or engine.scheduler.get_queue_size() > 0:
            engine.step()
            
        completed_requests.append(engine.completed_requests[0])
        
    total_duration = time.time() - start_time
    print(f"Sequential benchmark finished in {total_duration:.2f}s")
    return completed_requests

def run_continuous_batching(prompts: List[str], max_batch_size: int = 4) -> List[Request]:
    print(f"\n--- Running Continuous Batching Benchmark (Max Batch Size = {max_batch_size}) ---")
    
    engine = LLMEngine(max_batch_size=max_batch_size)
    for prompt in prompts:
        engine.add_request(prompt, max_tokens=MAX_TOKENS)
        
    start_time = time.time()
    step_count = 0
    while len(engine.scheduler.active_requests) > 0 or engine.scheduler.get_queue_size() > 0:
        engine.step()
        step_count += 1
        
    total_duration = time.time() - start_time
    print(f"Continuous batching benchmark finished in {total_duration:.2f}s (Total steps: {step_count})")
    return engine.completed_requests

def print_results(mode_name: str, requests: List[Request], benchmark_duration: float):
    num_reqs = len(requests)
    if num_reqs == 0:
        return
        
    total_tokens = sum(len(r.generated_tokens) for r in requests)
    avg_ttft = sum(r.ttft for r in requests if r.ttft is not None) / num_reqs
    avg_latency = sum(r.total_latency for r in requests if r.total_latency is not None) / num_reqs
    avg_tokens_sec = sum(r.tokens_per_sec for r in requests) / num_reqs
    system_throughput = total_tokens / benchmark_duration

    print(f"\n==========================================")
    print(f" RESULTS: {mode_name}")
    print(f"==========================================")
    print(f"Number of Requests:   {num_reqs}")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Benchmark Duration:    {benchmark_duration:.2f} seconds")
    print(f"System Throughput:     {system_throughput:.2f} tokens/sec")
    print(f"Avg TTFT:              {avg_ttft:.4f} seconds")
    print(f"Avg Latency:           {avg_latency:.4f} seconds")
    print(f"Avg Speed per Req:     {avg_tokens_sec:.2f} tokens/sec")
    print(f"==========================================")

def main():
    print("Initializing benchmark with SmolLM2-135M...")
    
    # Run sequential
    t_seq_start = time.time()
    seq_requests = run_sequential(TEST_PROMPTS)
    t_seq_duration = time.time() - t_seq_start
    
    # Run continuous batching
    t_cb_start = time.time()
    cb_requests = run_continuous_batching(TEST_PROMPTS, max_batch_size=4)
    t_cb_duration = time.time() - t_cb_start
    
    # Print comparison
    print_results("SEQUENTIAL", seq_requests, t_seq_duration)
    print_results("CONTINUOUS BATCHING", cb_requests, t_cb_duration)
    
    # Summary statement
    throughput_improvement = (
        (cb_requests[0].tokens_per_sec / seq_requests[0].tokens_per_sec - 1) * 100 
        if seq_requests[0].tokens_per_sec > 0 else 0
    )
    total_time_saved = t_seq_duration - t_cb_duration
    print(f"\nSummary: Continuous batching completed all tasks {total_time_saved:.2f}s faster than sequential.")
    print("Continuous batching allows overlapping prefill and decode stages, utilizing parallel hardware efficiently.")

if __name__ == "__main__":
    main()
