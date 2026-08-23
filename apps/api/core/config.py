"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings shared by the API and retrieval services.

    Environment variable names are derived from the field names, so for
    example ``search_timeout_seconds`` is read from ``SEARCH_TIMEOUT_SECONDS``.
    An optional local ``.env`` file is supported for development; secrets must
    not be committed to the repository.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    api_host: str = "0.0.0.0"
    # Render exposes its assigned port as PORT. API_PORT remains convenient
    # for local development and takes precedence when both are present.
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("API_PORT", "PORT"),
    )

    model_name: str = "distilbert"
    model_path: Path = Path("models/production")
    classical_model_path: Path = Path("models/classical/model.joblib")
    # Optional deployment-time download settings for a versioned model
    # artifact (for example, a GitHub Release asset).
    model_artifact_url: str | None = None
    model_artifact_sha256: str | None = None
    model_artifact_token: SecretStr | None = None
    min_article_length: int = Field(default=100, ge=1)
    max_article_length: int = Field(default=20_000, ge=1)

    # The web app owns NEXT_PUBLIC_API_URL; the API only needs the allowed
    # browser origins for CORS.
    cors_origins: str = ""

    # Current-source retrieval settings. The provider adapter is selected by
    # name so the API does not depend on a particular search vendor.
    search_provider: str | None = None
    search_endpoint: str | None = None
    search_api_key: str | None = None
    search_max_results: int = Field(default=10, ge=1, le=100)
    search_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    evidence_max_sources: int = Field(default=5, ge=1, le=50)
    evidence_max_claims: int = Field(default=5, ge=1, le=50)
    evidence_recency_days: int | None = Field(default=30, ge=0)

    @property
    def cors_origin_list(self) -> list[str]:
        """Return normalized origins for FastAPI's CORS middleware."""

        return [
            origin.strip().rstrip("/")
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @model_validator(mode="after")
    def validate_article_lengths(self) -> Settings:
        """Reject an impossible article-length range at startup."""

        if self.min_article_length > self.max_article_length:
            raise ValueError(
                "MIN_ARTICLE_LENGTH must not exceed MAX_ARTICLE_LENGTH"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings object for the process lifetime."""

    return Settings()


settings = get_settings()

__all__ = ["Settings", "get_settings", "settings"]
