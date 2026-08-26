# Changelog

The application version lives in `APP_VERSION` in `backend/main.py`.

## Unreleased

- Codebase split from RegCheck (July 2026): PreCheck is now its own
  repository serving only the registration-quality tool. Inherits the shared
  engine (retrieval, judgement, verification loop, evidence tracing, cost
  tracking) and the platform hardening from RegCheck v1.0.0.
