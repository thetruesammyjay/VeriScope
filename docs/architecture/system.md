
# Evidence-aware architecture

The system has two complementary online paths. The trained classifier
estimates whether article language resembles the labelled real or fake
examples. The evidence path extracts checkable claims, searches current
public sources, filters and ranks candidate documents, extracts relevant
passages, and compares those passages with each claim.

```text
article
  ├──> classifier ───────────────> likely-real / likely-fake + confidence
  └──> claim extraction
          └──> query builder
                  └──> search client
                          └──> source filter and document fetcher
                                  └──> passage ranker
                                          └──> claim verifier
                                                  └──> evidence summary
```

Search results are evidence candidates, not automatic truth. The response
must preserve source URLs, publication dates where available, retrieval time,
and an insufficient-evidence outcome.
