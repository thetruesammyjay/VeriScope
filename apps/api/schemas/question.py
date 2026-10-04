"""Contracts for source-grounded questions about news claims."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClaimQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class ClaimQuestionSource(BaseModel):
    source_id: str
    title: str
    url: str
    excerpt: str
    relevance_score: float = Field(ge=0, le=1)
    source_name: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None


class ClaimQuestionResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence", "answer_unavailable"]
    answer: str | None = None
    answer_engine: str | None = None
    cited_source_ids: list[str] = Field(default_factory=list)
    sources: list[ClaimQuestionSource] = Field(default_factory=list)
    message: str
    caution: str = (
        "AI-generated answers summarize retrieved material; they are not "
        "independent fact verification."
    )


__all__ = ["ClaimQuestionRequest", "ClaimQuestionResponse", "ClaimQuestionSource"]
