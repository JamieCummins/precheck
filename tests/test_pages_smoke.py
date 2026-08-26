"""Render every public PreCheck page once, and assert the comparison-suite
surfaces from the shared engine's ancestry are NOT served here."""
from __future__ import annotations

import os

import pytest


class FakeRedis:
    async def ping(self):
        return True

    async def hgetall(self, k):
        return {}

    async def get(self, k):
        return None

    async def exists(self, k):
        return 0


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pages")
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp}/pages.db"
    os.environ["SESSION_SECRET"] = "pages-secret"
    import backend.core.config as cfg

    cfg.get_settings.cache_clear()
    from starlette.testclient import TestClient

    from backend import create_app

    app = create_app()
    app.state.redis = FakeRedis()
    with TestClient(app) as c:
        yield c
    cfg.get_settings.cache_clear()


@pytest.mark.parametrize(
    "path",
    ["/", "/evaluate_registration", "/contact", "/team", "/privacy", "/login"],
)
def test_page_renders(client, path):
    resp = client.get(path)
    assert resp.status_code == 200, f"{path} -> {resp.status_code}"
    assert "<html" in resp.text.lower()


def test_precheck_branding(client):
    home = client.get("/").text
    assert "PreCheck" in home
    assert "RegCheck" not in home


@pytest.mark.parametrize(
    "path",
    ["/compare", "/demo", "/faq", "/api", "/docs", "/jobs", "/clinical_trials", "/general_preregistration"],
)
def test_comparison_surfaces_absent(client, path):
    assert client.get(path).status_code == 404
