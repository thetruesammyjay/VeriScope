"""Extract checkable claims from article text."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    """A factual statement selected for external verification."""

    claim_id: str
    text: str
    article_span: tuple[int, int] | None = None


def extract_claims(text: str, *, max_claims: int = 5) -> list[Claim]:
    """Return sentence-level factual claim candidates.

    The baseline favours sentences containing a factual verb, number, date,
    or attribution cue. It is intentionally conservative and preserves source
    offsets so a learned extractor can be substituted later.
    """

    if max_claims <= 0:
        return []

    claims: list[Claim] = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text, flags=re.DOTALL):
        sentence = " ".join(match.group(0).split())
        if not _is_checkable(sentence):
            continue
        start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        end = start + len(sentence)
        claims.append(
            Claim(
                claim_id=f"claim-{len(claims) + 1:03d}",
                text=sentence,
                article_span=(start, end),
            )
        )
        if len(claims) >= max_claims:
            break
    return claims


_FACTUAL_CUES = re.compile(
    r"\b(is|are|was|were|has|have|had|caused|contains|created|died|"
    r"increased|decreased|occurred|reported|announced|won|lost|according)\b",
    flags=re.IGNORECASE,
)


def _is_checkable(sentence: str) -> bool:
    words = re.findall(r"\b\w+\b", sentence)
    if len(words) < 5 or sentence.rstrip().endswith("?"):
        return False
    return bool(_FACTUAL_CUES.search(sentence) or re.search(r"\d", sentence))
