"""Transformer-only removal of known wire-service style markers.

The rules are intentionally boundary-only. Publisher or agency names in an
article's actual prose remain untouched; only datelines/bylines at the start
or a source signature at the end are removed.
"""

from __future__ import annotations

import re

_AGENCY = (
    r"Reuters|Associated\s+Press|The\s+Associated\s+Press|AP|AFP|"
    r"Agence\s+France-Presse|UPI|United\s+Press\s+International"
)
_PREFIX_PATTERNS = (
    re.compile(rf"^\s*(?:[A-Z][A-Z .,'-]{{2,40}}\s+)?\(({_AGENCY})\)\s*[-:–—]\s*", re.I),
    re.compile(rf"^\s*(?:{_AGENCY})\s*[-:–—]\s*", re.I),
    re.compile(rf"^\s*By\s+(?:{_AGENCY})(?:\s*[,.:–—-]\s*)", re.I),
)
_SUFFIX_PATTERN = re.compile(rf"\s*(?:\(({_AGENCY})\)|(?:[-–—|]\s*)?({_AGENCY}))\s*$", re.I)


def clean_article_text(text: str) -> str:
    """Strip a recognized leading agency dateline/byline and trailing credit."""

    cleaned = " ".join(str(text).split())
    for pattern in _PREFIX_PATTERNS:
        cleaned = pattern.sub("", cleaned, count=1)
    cleaned = _SUFFIX_PATTERN.sub("", cleaned, count=1)
    return cleaned.strip()


__all__ = ["clean_article_text"]
