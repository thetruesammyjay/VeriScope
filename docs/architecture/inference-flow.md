
# Evidence-aware inference flow

1. Validate article length and request format.
2. Run the selected production classifier.
3. Extract a bounded set of checkable claims.
4. Build focused queries for each claim.
5. Search the configured current-source provider.
6. Apply source policy and fetch permitted public documents.
7. Extract and rank relevant evidence passages.
8. Compare each claim with its evidence.
9. Return classification and evidence findings separately.

The system may return `mixed` or `insufficient` evidence. It must not force a
binary factual verdict when sources conflict or no adequate source is found.
