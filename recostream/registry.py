"""Model artifact registry on S3 (AWS S3, or any S3-compatible store such as
MinIO when S3_ENDPOINT is set).

Layout:
    s3://{bucket}/models/{version}/model.npz       embeddings + item features
    s3://{bucket}/models/{version}/ranker.pt       ranker weights
    s3://{bucket}/models/{version}/metrics.json    offline evaluation
    s3://{bucket}/models/production.json           {"version": ...} pointer

A new version is promoted to production only if it passes the evaluation
gate (NDCG@10 not worse than the current production model by > tolerance).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from .config import settings


def client():
    kw = {"region_name": settings.aws_region}
    if settings.s3_endpoint:
        kw["endpoint_url"] = settings.s3_endpoint
    return boto3.client("s3", **kw)


def ensure_bucket(s3=None) -> None:
    s3 = s3 or client()
    try:
        s3.head_bucket(Bucket=settings.s3_bucket)
    except ClientError:
        s3.create_bucket(Bucket=settings.s3_bucket)


def new_version() -> str:
    return time.strftime("v%Y%m%d-%H%M%S")


def upload(version: str, files: dict[str, Path], metrics: dict, s3=None) -> None:
    s3 = s3 or client()
    ensure_bucket(s3)
    for name, path in files.items():
        s3.upload_file(str(path), settings.s3_bucket, f"models/{version}/{name}")
    s3.put_object(Bucket=settings.s3_bucket, Key=f"models/{version}/metrics.json",
                  Body=json.dumps(metrics, indent=2).encode())


def production(s3=None) -> dict | None:
    s3 = s3 or client()
    try:
        obj = s3.get_object(Bucket=settings.s3_bucket, Key="models/production.json")
    except ClientError:
        return None
    return json.loads(obj["Body"].read())


def metrics_of(version: str, s3=None) -> dict:
    s3 = s3 or client()
    obj = s3.get_object(Bucket=settings.s3_bucket, Key=f"models/{version}/metrics.json")
    return json.loads(obj["Body"].read())


def promote_if_better(version: str, metrics: dict, tolerance: float = 0.005, s3=None) -> bool:
    """Evaluation gate. Returns True if `version` became production."""
    s3 = s3 or client()
    cur = production(s3)
    new_ndcg = metrics["ranker"]["ndcg@10"]
    if cur is not None:
        cur_ndcg = metrics_of(cur["version"], s3)["ranker"]["ndcg@10"]
        if new_ndcg < cur_ndcg - tolerance:
            print(f"[registry] {version} rejected: NDCG@10 {new_ndcg:.4f} < production {cur_ndcg:.4f}")
            return False
    s3.put_object(Bucket=settings.s3_bucket, Key="models/production.json",
                  Body=json.dumps({"version": version, "promoted_at": time.time(), "ndcg@10": new_ndcg}).encode())
    print(f"[registry] {version} promoted to production (NDCG@10 {new_ndcg:.4f})")
    return True


def download(version: str, dest: Path, s3=None) -> Path:
    s3 = s3 or client()
    out = dest / version
    out.mkdir(parents=True, exist_ok=True)
    for name in ("model.npz", "ranker.pt", "metrics.json"):
        target = out / name
        if not target.exists():
            s3.download_file(settings.s3_bucket, f"models/{version}/{name}", str(target))
    return out


def list_versions(s3=None) -> list[str]:
    s3 = s3 or client()
    resp = s3.list_objects_v2(Bucket=settings.s3_bucket, Prefix="models/", Delimiter="/")
    return sorted(p["Prefix"].split("/")[1] for p in resp.get("CommonPrefixes", []))
