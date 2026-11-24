"""Model registry: versioned uploads and the evaluation gate (in-process S3 mock)."""
import os

import pytest

pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402


@pytest.fixture
def registry(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.delenv("S3_ENDPOINT", raising=False)
    with mock_aws():
        import importlib

        from recostream import config, registry as reg
        importlib.reload(config)
        importlib.reload(reg)
        yield reg


def metrics(ndcg):
    return {"ranker": {"ndcg@10": ndcg}}


def test_gate_promotes_better_and_rejects_worse(registry, tmp_path):
    f = tmp_path / "model.npz"
    f.write_bytes(b"x")
    registry.upload("v1", {"model.npz": f}, metrics(0.080))
    assert registry.promote_if_better("v1", metrics(0.080))
    assert registry.production()["version"] == "v1"

    registry.upload("v2", {"model.npz": f}, metrics(0.090))
    assert registry.promote_if_better("v2", metrics(0.090))
    assert registry.production()["version"] == "v2"

    registry.upload("v3", {"model.npz": f}, metrics(0.070))  # regression
    assert not registry.promote_if_better("v3", metrics(0.070))
    assert registry.production()["version"] == "v2"

    assert registry.list_versions() == ["v1", "v2", "v3"]
    assert registry.metrics_of("v3")["ranker"]["ndcg@10"] == 0.070
