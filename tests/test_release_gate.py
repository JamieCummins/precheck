"""Release-gate surface: the /ready readiness probe (PreCheck has no public
API/docs surface)."""
from __future__ import annotations

import os

import pytest


class _FakeRedis:
    def __init__(self, *, ping_ok=True, heartbeats=1):
        self.ping_ok = ping_ok
        self.heartbeats = heartbeats

    async def ping(self):
        if not self.ping_ok:
            raise ConnectionError("redis down")
        return True

    async def keys(self, pattern):
        assert pattern == "worker:heartbeat:*"
        return [f"worker:heartbeat:{i}" for i in range(self.heartbeats)]

    async def hgetall(self, k):
        return {}

    async def get(self, k):
        return None


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("release")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/release.db"
    os.environ["SESSION_SECRET"] = "release-secret"
    import backend.core.config as cfg

    cfg.get_settings.cache_clear()
    from starlette.testclient import TestClient

    from backend import create_app

    app = create_app()
    app.state.redis = _FakeRedis()
    with TestClient(app) as c:
        yield c, app
    cfg.get_settings.cache_clear()


# ── /ready ─────────────────────────────────────────────────────────────────────


def test_ready_ok_when_all_dependencies_up(client):
    c, app = client
    app.state.redis = _FakeRedis(ping_ok=True, heartbeats=2)
    resp = c.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["worker"].startswith("ok")


def test_ready_503_without_worker_heartbeat(client):
    c, app = client
    app.state.redis = _FakeRedis(ping_ok=True, heartbeats=0)
    resp = c.get("/ready")
    assert resp.status_code == 503
    assert "no live worker heartbeat" in resp.json()["checks"]["worker"]


def test_ready_503_when_redis_down_and_no_internals_leaked(client):
    c, app = client
    app.state.redis = _FakeRedis(ping_ok=False)
    resp = c.get("/ready")
    assert resp.status_code == 503
    checks = resp.json()["checks"]
    assert checks["redis"] == "error: ConnectionError"  # class name only, no detail
    assert checks["worker"] == "unknown: redis unavailable"
    assert "redis down" not in resp.text  # exception text must not leak


def test_health_stays_dependency_free(client):
    c, app = client
    app.state.redis = _FakeRedis(ping_ok=False)  # even with redis down...
    assert c.get("/health").json() == {"status": "ok"}  # ...liveness is green
