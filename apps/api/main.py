"""FastAPI application bootstrap for the API service."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.api.routes.analyze import router as analyze_router
from apps.api.api.routes.health import router as health_router
from apps.api.core.config import Settings, get_settings


def create_app(app_settings: Settings | None = None) -> FastAPI:
    """Create the API application with environment-driven configuration."""

    resolved_settings = app_settings or get_settings()
    application = FastAPI(
        title="Automated Fake News Detection API",
        description="NLP-based classification and evidence-aware analysis.",
        version="0.1.0",
    )
    application.state.settings = resolved_settings

    if resolved_settings.cors_origin_list:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.cors_origin_list,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Keep the health endpoint outside /api/v1 so Render can use it as a
    # platform health check without depending on feature routes.
    application.include_router(health_router)
    application.include_router(analyze_router)
    return application


app = create_app()

__all__ = ["app", "create_app"]
