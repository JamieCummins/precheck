# PreCheck

AI-assisted evaluation of study-registration quality: PreCheck assesses how
completely and unambiguously a preregistration/registration specifies the
planned study — element by element, with every judgement traced to quoted
evidence in the source document. FastAPI serves the web UI; a Redis-backed
worker runs assessments; a CLI enables headless single runs and batch scoring.

PreCheck shares its engine with the open-source
[RegCheck](https://regcheck.app) project, from which it was split in July 2026.

Status: pre-launch. The assessment prompt and default quality criteria are
still being finalised and may change.

## Quick start
- Use `.venv/bin/python`; tests: `.venv/bin/python -m pytest -q`
- Headless run: `.venv/bin/python -m backend.cli quality --preregistration reg.pdf`
- Batch scoring: `.venv/bin/python -m backend.cli batch --manifest jobs.csv --output-dir out/`
- Deploy: see docs/precheck-deploy.md

## Layout
- `backend/` — FastAPI app, routes, services (assessment engine in
  `services/registration_quality.py`, shared retrieval/judgement machinery in
  `services/comparisons.py`), Redis worker (`worker.py`), CLI (`cli.py`).
- `templates/` + `static/` — frontend pages and assets.
- `migrations/` — Alembic schema (accounts, report ownership, sharing).
