"""Evaluate Registration Quality: pipeline, prompt, verification loop, orchestrator,
and route wiring (single-document flow — no paper anywhere)."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os

import numpy as np
import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("CLAUDE_API_KEY", "test")

from backend.services import comparisons  # noqa: E402
from backend.services import registration_quality as rq  # noqa: E402
from backend.services.dimensions import registration_quality_dimensions  # noqa: E402
from backend.services.embeddings import EmbeddingCorpus  # noqa: E402


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── defaults ───────────────────────────────────────────────────────────────────


def test_quality_default_dimensions_load():
    # Psychology is the default criteria set: since the 2026-08-21 codebook
    # alignment it mirrors the human evaluator worksheet — 8 dimensions, in
    # worksheet order, with "Hypothesis specificity" as the deliberate legacy
    # exception in slot 1.
    dims = registration_quality_dimensions()
    names = [d["dimension"] for d in dims]
    assert names == [
        "Hypothesis specificity",
        "Sample size or other stopping rules",
        "Assumption checks and violations",
        "Exclusion of observations, handling outliers and missing data",
        "Variable specification and preprocessing",
        "Manipulation checks",
        "Statistical method specification",
        "Inference",
    ]
    assert all((d.get("definition") or "").strip() for d in dims)


def test_quality_criteria_sets_selectable():
    from backend.services.dimensions import registration_quality_set_for_ui

    sets = registration_quality_set_for_ui()
    keys = [s["key"] for s in sets]
    # Psychology (the default) is listed first; clinical medicine is selectable.
    assert keys[0] == "psychology"
    assert "clinical" in keys
    # Psychology mirrors the 8-dimension human codebook; clinical (proposed
    # wording) still has its own 9.
    by_key = {s["key"]: len(s["dims"]) for s in sets}
    assert by_key["psychology"] == 8
    assert by_key["clinical"] == 9
    # An unknown set key falls back to the psychology defaults.
    assert registration_quality_dimensions("nonexistent") == registration_quality_dimensions()


def test_quality_set_not_in_comparison_disciplines():
    from backend.services.dimensions import discipline_keys

    assert "registration_quality" not in discipline_keys()


def test_normalize_quality_verdict():
    assert rq.normalize_quality_verdict("complete") == "complete"
    assert rq.normalize_quality_verdict("Fully specified") == "complete"
    assert rq.normalize_quality_verdict("PARTIAL") == "partial"
    assert rq.normalize_quality_verdict("absent") == "absent"
    # Current decision-criteria labels map onto the wire tokens; "no revision"
    # must win over the "revision required" substring it contains.
    assert rq.normalize_quality_verdict("present and no revision required") == "complete"
    assert rq.normalize_quality_verdict("Present and revision required") == "partial"
    assert rq.normalize_quality_verdict("not present") == "absent"
    assert rq.normalize_quality_verdict("") == "absent"
    assert rq.normalize_quality_verdict(None) == "absent"


# ── run_quality_assessment ────────────────────────────────────────────────────


def _corpus_cache():
    # Plant the "gamma criterion" chunk at index 5 so neighbour expansion
    # (window 2) around the retrieved PREREG_0001 cannot reach it — only the
    # targeted verification search can surface it.
    segments = [
        "we will recruit exactly 300 participants",
        "filler methods text",
        "filler design text",
        "filler measures text",
        "filler procedure text",
        "the gamma criterion is prespecified here",
    ]
    vecs = [[1.0, 0.0]] + [[0.9701, -0.2425]] * 4 + [[0.0, 1.0]]
    corpus = EmbeddingCorpus(
        segments=segments,
        embeddings=np.array(vecs, dtype=np.float32),
        chunk_ids=[f"PREREG_{i:04d}" for i in range(1, 7)],
        norms=np.array([1.0] * 6, dtype=np.float32),
        metadata=[{}] * 6,
    )
    return {f"prereg:{hashlib.sha256(b'p').hexdigest()}": corpus}


def _embed(text, model=None):
    if "gamma criterion" in text:
        return np.array([0.0, 1.0], dtype=np.float32)
    if "Sample size" in text:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def _reply(judgement="complete", rationale="Target N stated precisely [PREREG_0001].", quotes="[PREREG_0001]", unlocated=""):
    return json.dumps(
        {
            "dimension": "Sample size",
            "registration_content_quotes": quotes,
            "completeness_judgement": judgement,
            "completeness_rationale": rationale,
            "unlocated_in_registration": unlocated,
        }
    )


def test_quality_prompt_is_single_document_and_fields_map(monkeypatch):
    captured = []

    def _dispatch(messages, **_kw):
        captured.append(messages[-1]["content"])
        return _reply()

    monkeypatch.setattr(comparisons, "get_embedding", _embed)
    monkeypatch.setattr(comparisons, "_dispatch_judgement", _dispatch)

    result = rq.run_quality_assessment("p", "openai", "Sample size", top_k=1, corpus_cache=_corpus_cache())
    prompt = captured[0]
    # Single-document framing: registration excerpts only, no paper anywhere.
    assert "Registration excerpts:" in prompt
    assert "Paper excerpts:" not in prompt
    assert "completeness_judgement" in prompt
    # No comparison-flow schema leaks into the quality prompt.
    assert "deviation_judgement" not in prompt
    assert "unlocated_in_paper" not in prompt
    item = result.items[0]
    # Quality field aliases land on the canonical ComparisonItem fields.
    assert item.deviation_judgement == "complete"
    assert "Target N stated precisely" in item.deviation_information
    # Quote cards filtered to the cited excerpt; paper side empty.
    assert "PREREG_0001" in item.registration_content_quotes
    assert "PREREG_0002" not in item.registration_content_quotes
    assert item.paper_content_quotes == ""
    assert item.paper_content_summary == ""


def test_quality_custom_doctrine_and_context_capture(monkeypatch):
    captured = []

    def _dispatch(messages, **_kw):
        captured.append(messages[-1]["content"])
        return _reply()

    monkeypatch.setattr(comparisons, "get_embedding", _embed)
    monkeypatch.setattr(comparisons, "_dispatch_judgement", _dispatch)

    context: dict[str, str] = {}
    rq.run_quality_assessment(
        "p",
        "openai",
        "Sample size",
        top_k=1,
        corpus_cache=_corpus_cache(),
        custom_doctrine="MY CUSTOM DOCTRINE.\n",
        context_out=context,
    )
    prompt = captured[0]
    # The custom doctrine replaces the default static prefix entirely.
    assert prompt.startswith("MY CUSTOM DOCTRINE.")
    assert "Critically evaluate the completeness" not in prompt
    # The full judge context is captured per dimension for the CLI text column.
    assert "Sample size" in context
    assert "PREREG_0001" in context["Sample size"]


def test_cli_quality_csv_columns(tmp_path):
    import csv as csv_mod

    from backend import cli

    payload = {
        "items": [
            {
                "dimension": "Sample size",
                "text": "[PREREG_0001] Target N is 120.",
                "deviation_judgement": "Fully specified",
                "deviation_information": "Precise N stated.",
                "registration_content_quotes": "[PREREG_0001] Target N is 120.",
                "unlocated_in_registration": "",
            },
            {
                "dimension": "Exclusion of observations, handling outliers and missing data",
                "text": "[PREREG_0002] Exclusions listed.",
                "deviation_judgement": "partial",
                "deviation_information": "Thresholds vague.",
                "registration_content_quotes": "[PREREG_0002] Exclusions listed.",
                "unlocated_in_registration": "attention checks",
            },
        ],
        "cost": {"total_usd": 0.1234},
        "quality_meta": {
            "model_name": "gpt-5.5",
            "prompt_type": "default",
            "prompt_focus": "per_dimension",
            "retrieval": "rag",
            "chunk_size": 120,
            "price_in_usd": 0.1234,
        },
    }
    out = tmp_path / "quality.csv"
    cli._write_output(payload, str(out), "csv")

    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv_mod.DictReader(fh))
    assert list(rows[0].keys()) == [
        "model_name",
        "prompt_type",
        "prompt_focus",
        "retrieval",
        "chunk_size",
        "price_in_usd",
        "dimension",
        "text",
        "evaluation",
        "justification",
        "registration_content_quotes",
        "unlocated_in_registration",
    ]
    assert len(rows) == 2
    # Verdicts emit the decision-criteria labels; legacy labels normalize down.
    assert rows[0]["evaluation"] == "present and no revision required"
    assert rows[1]["evaluation"] == "present and revision required"
    # Run-level values repeat on every row (long-format analysis table).
    assert rows[0]["price_in_usd"] == rows[1]["price_in_usd"] == "0.1234"
    assert rows[0]["model_name"] == "gpt-5.5"
    assert rows[1]["justification"] == "Thresholds vague."


def test_quality_verification_loop_augments_and_rejudges(monkeypatch):
    calls = []

    def _dispatch(messages, **_kw):
        calls.append(messages[-1]["content"])
        if len(calls) == 1:
            return _reply(judgement="partial", unlocated="gamma criterion")
        return _reply(judgement="complete", rationale="Found the gamma criterion [PREREG_0006].")

    monkeypatch.setattr(comparisons, "get_embedding", _embed)
    monkeypatch.setattr(comparisons, "_dispatch_judgement", _dispatch)

    result = rq.run_quality_assessment("p", "openai", "Sample size", top_k=1, corpus_cache=_corpus_cache())
    assert len(calls) == 2
    assert "Additional targeted registration excerpts" in calls[1]
    assert "gamma criterion" in calls[1]
    item = result.items[0]
    assert item.deviation_judgement == "complete"  # re-judgement wins
    assert "the assessment above includes them" in item.deviation_information


def test_quality_no_rag_shows_whole_document_and_skips_retrieval(monkeypatch):
    calls = []

    def _dispatch(messages, **_kw):
        calls.append(messages[-1]["content"])
        # Unlocated element would trigger the verification loop under RAG.
        return _reply(judgement="partial", unlocated="gamma criterion")

    def _no_embedding(*_a, **_k):
        raise AssertionError("no embedding call should happen with use_rag=False")

    monkeypatch.setattr(comparisons, "get_embedding", _no_embedding)
    monkeypatch.setattr(comparisons, "_dispatch_judgement", _dispatch)

    result = rq.run_quality_assessment(
        "p", "openai", "Sample size", corpus_cache=_corpus_cache(), use_rag=False
    )
    # One judgement only: no targeted verification loop in full-document mode.
    assert len(calls) == 1
    prompt = calls[0]
    # Every chunk, document order, no similarity labels.
    for i in range(1, 7):
        assert f"[PREREG_{i:04d}]" in prompt
    assert "relevance_score=" not in prompt
    assert prompt.index("[PREREG_0001]") < prompt.index("[PREREG_0006]")
    # Full-document doctrine paragraph swapped in, RAG paragraph out.
    assert "nothing has been filtered out" in prompt
    assert "You see only retrieved excerpts" not in prompt
    item = result.items[0]
    assert item.deviation_judgement == "partial"
    # No "targeted full-document search" note appended.
    assert "targeted full-document search" not in item.deviation_information


def test_quality_no_rag_builds_chunk_only_corpus(monkeypatch):
    from backend.services import embeddings as emb

    def _no_embedding(*_a, **_k):
        raise AssertionError("no embedding call should happen with use_rag=False")

    monkeypatch.setattr(emb, "openai_embed_segments", _no_embedding)
    monkeypatch.setattr(comparisons, "get_embedding", _no_embedding)
    monkeypatch.setattr(comparisons, "_dispatch_judgement", lambda *a, **k: _reply())

    cache: dict = {}
    text = "We will recruit exactly 300 participants. " + "Filler methods sentence. " * 40
    result = rq.run_quality_assessment(
        text, "openai", "Sample size", corpus_cache=cache, use_rag=False
    )
    corpus = next(iter(cache.values()))
    assert corpus.segments  # chunked
    assert corpus.embeddings.size == 0  # but never embedded
    assert corpus.chunk_ids[0] == "PREREG_0001"
    assert result.items[0].deviation_judgement == "complete"


def test_quality_unparseable_reply_degrades(monkeypatch):
    monkeypatch.setattr(comparisons, "get_embedding", _embed)
    monkeypatch.setattr(comparisons, "_dispatch_judgement", lambda *a, **k: "not json at all")

    result = rq.run_quality_assessment("p", "openai", "Sample size", top_k=1, corpus_cache=_corpus_cache())
    item = result.items[0]
    assert "couldn’t parse" in item.deviation_information.lower() or "couldn't parse" in item.deviation_information.lower()


# ── orchestrator ──────────────────────────────────────────────────────────────


def test_orchestrator_runs_every_default_dimension(tmp_path):
    reg_path = tmp_path / "registration.txt"
    reg_path.write_text("We will recruit exactly 300 participants.", encoding="utf-8")

    seen = []

    def fake_runner(registration_input, client, dimension, **kwargs):
        seen.append((dimension, kwargs.get("dimension_definition", "")))
        return comparisons.ComparisonResult(
            items=[comparisons.ComparisonItem(dimension=dimension, deviation_judgement="partial")]
        )

    result = _run(
        rq.registration_quality_assessment(
            str(reg_path),
            ".txt",
            "openai",
            "pymupdf",
            assessment_runner=fake_runner,
        )
    )
    defaults = registration_quality_dimensions()
    assert len(seen) == len(defaults) == len(result.items)
    assert [s[0] for s in seen] == [d["dimension"] for d in defaults]
    assert all(defn.strip() for _n, defn in seen)  # definitions reach the runner


# ── route wiring ──────────────────────────────────────────────────────────────


class _FakeRedis:
    def __init__(self):
        self.h: dict[str, dict] = {}
        self.queue: list[str] = []
        self.kv: dict[str, str] = {}

    async def ping(self):
        return True

    async def hgetall(self, k):
        return self.h.get(k, {})

    async def hset(self, k, mapping=None):
        self.h.setdefault(k, {}).update(mapping or {})

    async def llen(self, k):
        return len(self.queue) if k == "comparison:queue" else 0

    async def rpush(self, k, value):
        self.queue.append(value)

    async def set(self, k, value, nx=False, ex=None):
        self.kv[k] = value
        return True

    async def expire(self, k, ttl):
        return True

    async def persist(self, k):
        return True

    async def get(self, k):
        return self.kv.get(k)

    async def exists(self, k):
        return 1 if k in self.h else 0


@pytest.fixture()
def client(tmp_path):
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path}/rq.db"
    os.environ["SESSION_SECRET"] = "rq-secret"
    os.environ["UPLOAD_DIR"] = str(tmp_path / "uploads")
    import backend.core.config as cfg

    cfg.get_settings.cache_clear()
    from starlette.testclient import TestClient

    from backend import create_app

    app = create_app()
    app.state.redis = _FakeRedis()
    with TestClient(app) as c:
        yield c, app
    cfg.get_settings.cache_clear()
    os.environ.pop("UPLOAD_DIR", None)


def test_evaluate_registration_post_needs_no_paper(client):
    c, app = client
    dims = json.dumps([{"dimension": "Sample size and stopping rule", "definition": "x"}])
    resp = c.post(
        "/evaluate_registration",
        data={
            "parser_choice": "pymupdf",
            "client": "openai",
            "prereg_source": "upload",
            "dimensions_data": dims,
        },
        files={"preregistration": ("reg.txt", io.BytesIO(b"We will recruit 300 participants."), "text/plain")},
        follow_redirects=False,
    )
    assert resp.status_code == 302, resp.text
    assert resp.headers["location"].startswith("/survey/")
    assert len(app.state.redis.queue) == 1
    job = json.loads(app.state.redis.queue[0])
    assert job["comparison_type"] == "registration_quality"
    assert "paper_path" not in job
    assert job["prereg_path"]
    # The stored registration blob is what the worker will restore.
    blob = app.state.redis.kv.get(f"upload:{job['task_id']}:prereg")
    assert blob and b"recruit 300" in __import__("gzip").decompress(base64.b64decode(blob))
    # Settings recorded for the viewer's flag-gated two-panel mode.
    settings = json.loads(app.state.redis.h[job["task_id"]]["settings_json"])
    assert settings["comparison_type"] == "registration_quality"
