"""Source-grounded question answering for news claims."""

from __future__ import annotations

from typing import Annotated

from apps.api.api.dependencies import get_claim_question_service
from apps.api.schemas.question import ClaimQuestionRequest, ClaimQuestionResponse
from apps.api.services.claim_question import ClaimQuestionService
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/api/v1", tags=["claim questions"])


@router.post("/ask", response_model=ClaimQuestionResponse)
def ask_about_claim(
    request: ClaimQuestionRequest,
    service: Annotated[ClaimQuestionService, Depends(get_claim_question_service)],
) -> ClaimQuestionResponse:
    """Search current sources and answer a news-claim question cautiously."""

    question = request.question.strip()
    if len(question) < 5:
        raise HTTPException(
            status_code=422,
            detail="Ask a question of at least 5 non-whitespace characters.",
        )
    return service.ask(question)


__all__ = ["ask_about_claim", "router"]
