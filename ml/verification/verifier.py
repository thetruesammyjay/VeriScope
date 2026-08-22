"""Compare claims with retrieved evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .claim_extractor import Claim
from .evidence_extractor import EvidencePassage

EvidenceStatus = Literal["supported", "contradicted", "mixed", "insufficient"]


@dataclass(frozen=True)
class ClaimAssessment:
    claim: Claim
    status: EvidenceStatus
    evidence: tuple[EvidencePassage, ...] = ()
    rationale: str | None = None


def verify_claim(
    claim: Claim,
    evidence: list[EvidencePassage],
) -> ClaimAssessment:
    """Return a claim assessment placeholder for the evidence verifier."""

    del evidence
    return ClaimAssessment(claim=claim, status="insufficient")

