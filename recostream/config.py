"""Runtime configuration, read from environment variables."""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("RECO_DATA_DIR", "data")))
    raw_dir: Path = field(default_factory=lambda: Path(_env("RECO_RAW_DIR", "../datasets/ml-25m")))
    artifacts_dir: Path = field(default_factory=lambda: Path(_env("RECO_ARTIFACTS", "artifacts")))

    kafka_bootstrap: str = field(default_factory=lambda: _env("KAFKA_BOOTSTRAP", "127.0.0.1:9092"))
    kafka_topic: str = field(default_factory=lambda: _env("KAFKA_TOPIC", "interactions"))

    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://127.0.0.1:6379/0"))

    s3_endpoint: str | None = field(default_factory=lambda: os.environ.get("S3_ENDPOINT"))  # None = real AWS S3
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "recostream-models"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))

    # Model hyper-parameters
    emb_dim: int = 64
    n_candidates: int = 200
    history_len: int = 50  # recent items kept per user in the online feature store
    k: int = 10


settings = Settings()
