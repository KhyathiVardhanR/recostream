"""FastAPI model-serving API.

    uvicorn recostream.serving.app:app --workers 4 --port 8000

Endpoints
    GET  /recommend/{user}?k=10   personalized top-k (dense user id)
    GET  /healthz
    GET  /model                   active model version + offline metrics
    POST /admin/reload            load the current production model from S3
    GET  /metrics                 Prometheus

Requests are micro-batched: concurrent requests arriving within a few
milliseconds share one Redis round trip, one candidate matmul and one ranker
forward pass (batch prediction), which keeps tail latency flat under load.
"""
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import ORJSONResponse
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from .. import feature_store, registry
from ..config import settings
from .recommender import Model, Recommender

torch.set_num_threads(int(os.environ.get("RECO_TORCH_THREADS", "2")))

MAX_BATCH = int(os.environ.get("RECO_MAX_BATCH", "64"))
MAX_WAIT_MS = float(os.environ.get("RECO_MAX_WAIT_MS", "3"))

LATENCY = Histogram("reco_request_seconds", "End-to-end /recommend latency",
                    buckets=(.005, .01, .015, .02, .03, .04, .05, .075, .1, .15, .2, .3, .5, 1))
STAGE = Histogram("reco_stage_seconds", "Per-batch stage latency", ["stage"],
                  buckets=(.0005, .001, .002, .005, .01, .02, .05, .1, .2))
BATCH = Histogram("reco_batch_size", "Users per inference batch", buckets=(1, 2, 4, 8, 16, 32, 64, 128))
SERVED = Counter("reco_recommendations_total", "Recommendation requests served")
MODEL_INFO = Gauge("reco_model_info", "Active model (value = offline NDCG@10)", ["version"])
ONLINE_HIT_RATE = Gauge("reco_online_hit_rate", "Share of streamed events whose item was in the served top-k")

app = FastAPI(title="RecoStream", default_response_class=ORJSONResponse)
app.mount("/metrics", make_asgi_app())

state: dict = {}
executor = ThreadPoolExecutor(max_workers=1)  # one inference thread per worker process


def load_model() -> Model:
    local = os.environ.get("RECO_MODEL_DIR")
    if local:
        return Model.load(Path(local))
    prod = registry.production()
    if not prod:
        raise RuntimeError("no production model in the registry; run `python -m recostream.train` first")
    path = registry.download(prod["version"], settings.artifacts_dir / "cache")
    return Model.load(path)


def install(model: Model) -> None:
    state["rec"] = Recommender(model, feature_store.client())
    MODEL_INFO.clear()
    MODEL_INFO.labels(model.version).set(model.metrics.get("ranker", {}).get("ndcg@10", 0))


class Batcher:
    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()

    async def submit(self, user: int, k: int):
        fut = asyncio.get_running_loop().create_future()
        await self.q.put((user, k, fut))
        return await fut

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.q.get()
            batch = [first]
            deadline = loop.time() + MAX_WAIT_MS / 1000
            while len(batch) < MAX_BATCH:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.q.get(), timeout))
                except asyncio.TimeoutError:
                    break
            users = [b[0] for b in batch]
            k = max(b[1] for b in batch)
            try:
                items, scores, timings = await loop.run_in_executor(executor, state["rec"].recommend_batch, users, k)
                BATCH.observe(len(batch))
                for stage, secs in timings.items():
                    STAGE.labels(stage).observe(secs)
                pipe = state["rec"].r.pipeline(transaction=False)
                for (u, kk, fut), it, sc in zip(batch, items, scores):
                    feature_store.record_served(pipe, u, it[:kk])
                    if not fut.done():
                        fut.set_result((it[:kk], sc[:kk]))
                loop.run_in_executor(None, pipe.execute)
            except Exception as e:  # noqa: BLE001
                for _, _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)


batcher = Batcher()


@app.on_event("startup")
async def startup():
    install(load_model())
    asyncio.create_task(batcher.run())


@app.get("/healthz")
async def healthz():
    return {"ok": True, "model": state["rec"].m.version}


@app.get("/model")
async def model_info():
    m = state["rec"].m
    return {"version": m.version, "metrics": m.metrics}


@app.post("/admin/reload")
async def reload():
    model = await asyncio.get_running_loop().run_in_executor(None, load_model)
    install(model)
    return {"version": model.version}


@app.get("/recommend/{user}")
async def recommend(user: int, k: int = Query(10, ge=1, le=100)):
    t0 = time.perf_counter()
    if user < 0:
        raise HTTPException(400, "user must be non-negative")
    items, scores = await batcher.submit(user, k)
    m = state["rec"].m
    SERVED.inc()
    LATENCY.observe(time.perf_counter() - t0)
    return {
        "user": user,
        "model": m.version,
        "items": [{"item": i, "movieId": int(m.item_ids[i]), "title": str(m.titles[i]), "score": s}
                  for i, s in zip(items, scores)],
    }


@app.get("/online-quality")
async def online_quality():
    r = state["rec"].r
    events, with_recs, hits = (int(x or 0) for x in r.mget("eval:events", "eval:events_with_recs", "eval:hits"))
    rate = hits / with_recs if with_recs else 0.0
    ONLINE_HIT_RATE.set(rate)
    return {"events": events, "events_with_recs": with_recs, "hits": hits, "online_hit_rate": rate}
