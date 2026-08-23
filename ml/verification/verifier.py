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
    """Classify evidence with a transparent contradiction-cue baseline."""

    if not evidence:
        return ClaimAssessment(
            claim=claim,
            status="insufficient",
            rationale="No relevant evidence passage was retrieved.",
        )

    support = False
    contradiction = False
    for passage in evidence:
        text = passage.text.lower()
        if any(cue in text for cue in _CONTRADICTION_CUES):
            contradiction = True
        else:
            support = True

    if support and contradiction:
        status: EvidenceStatus = "mixed"
        rationale = "Retrieved passages contain both supporting and contradicting cues."
    elif contradiction:
        status = "contradicted"
        rationale = "Retrieved passages contain an explicit contradiction cue."
    else:
        status = "supported"
        rationale = "Retrieved passages overlap the claim without a contradiction cue."
    return ClaimAssessment(
        claim=claim,
        status=status,
        evidence=tuple(evidence),
        rationale=rationale,
    )


_CONTRADICTION_CUES = (
    " is false",
    " are false",
    " was false",
    " were false",
    "not true",
    "incorrect",
    "inaccurate",
    "debunked",
    "refuted",
    "denied",
    "did not",
    "does not",
    "never happened",
)
