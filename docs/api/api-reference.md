
# Analysis API (classification plus current evidence)

The evidence-aware workflow will be exposed separately from the existing
classification endpoint:

```text
POST /api/v1/analyze
```

The request accepts article text. The response is designed to include the
classification label and confidence, extracted claims, evidence status,
source titles and URLs, publication dates when available, retrieval time,
model metadata, processing time, and a responsible-use disclaimer.

Evidence statuses are deliberately non-binary:

```text
supported | contradicted | mixed | insufficient
```

The endpoint will not be registered until a concrete search provider,
document fetcher, and verification model are configured.
