
# Analysis API (current-source evidence)

The first implemented evidence workflow is exposed at:

```text
POST /api/v1/analyze
```

The request accepts article text:

```json
{"text": "The city has 3 hospitals."}
```

The current response contains extracted claims, evidence passages, source
URLs, relevance scores, and claim-level statuses. Classifier metadata will be
added when the ML inference service is connected.

When no search provider is configured, the endpoint remains available and
returns `insufficient` rather than fabricating evidence.

Evidence statuses are deliberately non-binary:

```text
supported | contradicted | mixed | insufficient
```

Set `SEARCH_PROVIDER=bing`, `SEARCH_ENDPOINT`, and `SEARCH_API_KEY` to enable
the live provider adapter. The document fetcher and verification baseline are
provider-independent.
