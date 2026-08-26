from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..services.dimensions import registration_quality_set_for_ui

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "index_precheck.html")


@router.get("/evaluate_registration", response_class=HTMLResponse, name="evaluate_registration")
async def evaluate_registration(request: Request):
    templates = request.app.state.templates
    # The quality criteria are defined once in the backend and injected here so
    # the wizard and CLI resolve the same criteria/definitions.
    return templates.TemplateResponse(
        request,
        "registration_quality.html",
        {"discipline_sets_json": json.dumps(registration_quality_set_for_ui())},
    )


@router.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "contact.html")


@router.get("/team", name="team")
async def team(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "team.html")


@router.get("/privacy", response_class=HTMLResponse, name="privacy")
async def privacy(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "privacy.html")
