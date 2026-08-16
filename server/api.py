import asyncio
import os
import time
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.engine import LLMEngine
from engine.request import Request

app = FastAPI(title="tiny-vLLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

engine = LLMEngine(max_batch_size=4)
engine_task = None

async def run_engine_loop():
    print("[SERVER] Background LLM Engine loop started.")
    while True:
        try:
            has_work = engine.scheduler.get_queue_size() > 0 or len(engine.scheduler.active_requests) > 0
            if has_work:
                await asyncio.to_thread(engine.step)
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[SERVER] Engine loop error: {type(e).__name__}: {e}")
            await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    global engine_task
    engine_task = asyncio.create_task(run_engine_loop())

@app.on_event("shutdown")
async def shutdown_event():
    if engine_task:
        engine_task.cancel()

# ----------------------------------------------------------------- models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    priority: int = 0

class ConfigRequest(BaseModel):
    model_id: Optional[str] = None
    mode: Optional[str] = None
    scheduler: Optional[str] = None
    max_batch_size: Optional[int] = None
    precision: Optional[str] = None
    prefix_cache: Optional[bool] = None
    chunked_prefill: Optional[bool] = None
    chunk_size: Optional[int] = None
    speculative: Optional[bool] = None
    spec_tokens: Optional[int] = None
    draft_model_id: Optional[str] = None

class CompareRequest(BaseModel):
    prompts: Optional[List[str]] = None
    max_tokens: int = 20
    concurrency: int = 6

class SweepRequest(BaseModel):
    param: str            # "chunk_size" | "max_batch_size" | "concurrency"
    values: List[int]
    prompts: Optional[List[str]] = None
    max_tokens: int = 20
    concurrency: int = 4
    concurrency: int = 6

# ----------------------------------------------------------------- routes
@app.post("/generate")
async def generate(req: GenerateRequest):
    request_id = f"req_{int(time.time_ns())}"
    request = engine.add_request(prompt=req.prompt, max_tokens=req.max_tokens,
                                 temperature=req.temperature, top_p=req.top_p,
                                 top_k=req.top_k, priority=req.priority, request_id=request_id)
    while request.status != "FINISHED":
        await asyncio.sleep(0.05)
    return {
        "request_id": request.request_id,
        "prompt": req.prompt,
        "text": engine.model.decode_tokens(request.generated_tokens),
        "prefix_cache_hit": request.prefix_cache_hit,
        "metrics": {
            "ttft": request.ttft, "latency": request.total_latency,
            "tokens_sec": request.tokens_per_sec,
            "tokens_generated": len(request.generated_tokens)
        }
    }

@app.get("/stream")
async def stream_generate(prompt: str, max_tokens: int = Query(50),
                          temperature: float = Query(1.0), top_p: float = Query(1.0),
                          top_k: int = Query(50)):
    request_id = f"req_{int(time.time_ns())}"
    request = engine.add_request(prompt=prompt, max_tokens=max_tokens,
                                 temperature=temperature, top_p=top_p, top_k=top_k,
                                 request_id=request_id)

    async def sse_generator():
        last_idx = 0
        try:
            while request.status != "FINISHED" or last_idx < len(request.generated_tokens):
                while last_idx < len(request.generated_tokens):
                    token_id = request.generated_tokens[last_idx]
                    yield f"data: {engine.model.decode_token(token_id)}\n\n"
                    last_idx += 1
                await asyncio.sleep(0.03)
        except asyncio.CancelledError:
            request.status = "FINISHED"
            request.finish_time = time.time()
            raise
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.get("/metrics")
def get_metrics():
    return engine.get_metrics()

@app.get("/config")
def get_config():
    return engine.config

@app.post("/config")
def set_config(cfg: ConfigRequest):
    changes = {k: v for k, v in cfg.dict().items() if v is not None}
    return engine.apply_config(**changes)

@app.get("/timeseries")
def get_timeseries():
    return engine.history

@app.get("/gantt")
def get_gantt():
    return engine.get_gantt()

@app.get("/status")
def get_status():
    kv_per_tok = engine.model.kv_bytes_per_token()
    return {
        "waiting": [{"id": r.request_id, "prompt": r.prompt,
                     "seq_len": len(r.prompt_tokens), "priority": r.priority}
                    for r in engine.scheduler.waiting_queue],
        "active": [{
            "id": r.request_id, "prompt": r.prompt,
            "generated_count": len(r.generated_tokens), "max_tokens": r.max_tokens,
            "ttft": r.ttft, "start_time": r.start_time, "seq_len": r.seq_len,
            "status": r.status, "prefix_cache_hit": r.prefix_cache_hit,
            "prompt_kv_len": r.prompt_kv_len,
            "kv_bytes": r.prompt_kv_len * kv_per_tok,
        } for r in engine.scheduler.active_requests],
        "metrics": engine.get_metrics(),
        "completed": [{
            "id": r.request_id, "prompt": r.prompt,
            "generated": engine.model.decode_tokens(r.generated_tokens),
            "ttft": r.ttft, "latency": r.total_latency,
            "tokens_sec": r.tokens_per_sec, "tokens_count": len(r.generated_tokens),
            "prefix_cache_hit": r.prefix_cache_hit,
        } for r in engine.completed_requests[-8:]]
    }

# ----------------------------------------------------------------- compare benchmark
DEFAULT_PROMPTS = [
    "What is a GPU?",
    "Explain gravity in simple words.",
    "Write a short poem about the sea.",
    "Name three programming languages and their uses.",
    "What causes the seasons to change?",
    "Describe a black hole briefly.",
]

async def _run_workload(prompts: List[str], max_tokens: int) -> dict:
    """Fire a batch of concurrent requests, wait for completion, return aggregate metrics."""
    engine.completed_requests.clear()
    engine.history.clear()
    engine.step_count = 0
    engine.total_batch_size = 0
    engine.max_batch_size_seen = 0
    batch = [engine.add_request(p, max_tokens=max_tokens, temperature=0.7,
            request_id=f"w_{i}_{int(time.time_ns())}") for i, p in enumerate(prompts)]
    t0 = time.time()
    while any(r.status != "FINISHED" for r in batch):
        await asyncio.sleep(0.05)
    elapsed = time.time() - t0
    total_tokens = sum(len(r.generated_tokens) for r in batch)
    ttfts = [r.ttft for r in batch if r.ttft]
    lats = [r.total_latency for r in batch if r.total_latency]
    return {
        "total_time_s": round(elapsed, 2),
        "total_tokens": total_tokens,
        "throughput_tps": round(total_tokens / elapsed, 2) if elapsed > 0 else 0,
        "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else 0,
        "avg_latency_s": round(sum(lats) / len(lats), 3) if lats else 0,
        "max_batch_seen": engine.max_batch_size_seen,
        "memory_mb": round(engine._mem_mb(), 1),
    }

@app.post("/compare")
async def compare_modes(req: CompareRequest):
    """
    Run a fixed workload across all three serving modes (sequential, static,
    continuous) and return comparative metrics for each. Restores prior config.
    """
    prompts = req.prompts or DEFAULT_PROMPTS[:req.concurrency]
    max_tokens = req.max_tokens
    saved = dict(engine.config)

    # Disable prefix caching for a fair across-mode comparison (each mode faces cold prefill).
    engine.prefix_cache.clear()

    results = []
    for mode in ["sequential", "static", "continuous"]:
        engine.apply_config(mode=mode, max_batch_size=saved["max_batch_size"],
                            scheduler=saved["scheduler"], precision=saved["precision"],
                            prefix_cache=False)
        engine.prefix_cache.clear()
        stats = await _run_workload(prompts, max_tokens)
        stats["mode"] = mode
        results.append(stats)

    # restore (re-enable prefix cache to the saved setting)
    engine.apply_config(mode=saved["mode"], max_batch_size=saved["max_batch_size"],
                        scheduler=saved["scheduler"], precision=saved["precision"],
                        prefix_cache=saved["prefix_cache"], chunked_prefill=saved["chunked_prefill"],
                        chunk_size=saved["chunk_size"], speculative=saved["speculative"],
                        spec_tokens=saved["spec_tokens"])
    return {"workload": {"prompts": len(prompts), "max_tokens": max_tokens}, "results": results}

@app.post("/sweep")
async def sweep(req: SweepRequest):
    """
    Run a parameter sweep and return per-value metrics.
    Supported params: chunk_size, max_batch_size, concurrency.
    """
    prompts = req.prompts or DEFAULT_PROMPTS
    max_tokens = req.max_tokens
    saved = dict(engine.config)
    engine.prefix_cache.clear()
    results = []
    for v in req.values:
        engine.apply_config(mode="continuous", max_batch_size=saved["max_batch_size"],
                            scheduler=saved["scheduler"], precision=saved["precision"],
                            prefix_cache=False)
        if req.param == "chunk_size":
            engine.apply_config(chunked_prefill=True, chunk_size=v)
        elif req.param == "max_batch_size":
            engine.apply_config(max_batch_size=v)
        elif req.param == "concurrency":
            engine.apply_config(max_batch_size=saved["max_batch_size"])
        else:
            return {"error": f"unknown param {req.param}"}
        engine.prefix_cache.clear()
        use_prompts = prompts[:v] if req.param == "concurrency" else prompts
        stats = await _run_workload(use_prompts, max_tokens)
        stats["value"] = v
        results.append(stats)
    engine.apply_config(mode=saved["mode"], max_batch_size=saved["max_batch_size"],
                        scheduler=saved["scheduler"], precision=saved["precision"],
                        prefix_cache=saved["prefix_cache"], chunked_prefill=saved["chunked_prefill"],
                        chunk_size=saved["chunk_size"], speculative=saved["speculative"],
                        spec_tokens=saved["spec_tokens"])
    return {"param": req.param, "results": results}

@app.get("/")
def read_root():
    dash_path = "/Users/aryanlohia/Desktop/vllmcopy/dashboard/index.html"
    if os.path.exists(dash_path):
        return FileResponse(dash_path, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        })
    return {"message": "tiny-vLLM API is running. Dashboard file not found."}
