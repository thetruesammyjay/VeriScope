"""Aggregate claim-level assessments into an article-level finding."""

from __future__ import annotations

from dataclasses import dataclass

from .verifier import ClaimAssessment, EvidenceStatus


@dataclass(frozen=True)
class VerificationSummary:
    status: EvidenceStatus
    assessments: tuple[ClaimAssessment, ...] = ()


def aggregate_assessments(
    assessments: list[ClaimAssessment],
    *,
    transformer_retrieval: bool = False,
) -> VerificationSummary:
    """Combine claim findings without forcing unsupported binary certainty."""

    if not assessments:
        return VerificationSummary(status="insufficient")
    if transformer_retrieval:
        status: EvidenceStatus = (
            "sources_found"
            if any(assessment.evidence for assessment in assessments)
            else "insufficient"
        )
        return VerificationSummary(status=status, assessments=tuple(assessments))
    statuses = {assessment.status for assessment in assessments}
    if len(statuses) == 1:
        status = statuses.pop()
    elif "supported" in statuses and "contradicted" in statuses:
        status = "mixed"
    else:
        status = "mixed"
    return VerificationSummary(status=status, assessments=tuple(assessments))
