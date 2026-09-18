"""Schemas for combined classification and current-source analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .prediction import PredictionResponse
from .verification import VerificationResponse


class AnalysisRequest(BaseModel):
    """Raw submitted text; deployment-specific limits are checked by the route."""

    text: str = Field(min_length=1)


class AnalysisResponse(BaseModel):
    prediction: PredictionResponse
    verification: VerificationResponse


def validate_article_text(
    text: str,
    *,
    min_length: int,
    max_length: int,
) -> str:
    """Validate a submitted article against the active deployment limits.

    Whitespace-only submissions must not pass merely because they contain
    characters. The original text is returned so model preprocessing remains
    the single owner of normalisation.
    """

    article_length = len(text.strip())
    if article_length < min_length:
        raise ValueError(
            f"Article text must contain at least {min_length} non-whitespace characters."
        )
    if article_length > max_length:
        raise ValueError(
            f"Article text must not exceed {max_length} characters."
        )
    return text


__all__ = ["AnalysisRequest", "AnalysisResponse", "validate_article_text"]
