"""Feature construction for the classical text classifier."""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ClassicalConfig


def normalize_text(text: str) -> str:
    """Apply conservative normalization while retaining textual signals."""

    return " ".join(re.sub(r"\s+", " ", text.strip().lower()).split())


def build_vectorizer(config: ClassicalConfig | None = None) -> TfidfVectorizer:
    """Create the reproducible TF-IDF feature extractor."""

    resolved = config or ClassicalConfig()
    return TfidfVectorizer(
        lowercase=False,
        max_features=resolved.max_features,
        ngram_range=resolved.ngram_range,
        min_df=resolved.min_df,
        max_df=resolved.max_df,
        sublinear_tf=True,
    )


__all__ = ["build_vectorizer", "normalize_text"]
