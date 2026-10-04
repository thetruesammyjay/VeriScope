import type { ClaimQuestionResponse } from "@/lib/types";

const statusLabels = {
  answered: "Sources found",
  insufficient_evidence: "Not enough evidence",
  answer_unavailable: "Answer unavailable",
} as const;

function formatDate(value?: string | null) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleDateString();
}

export function ClaimAnswer({ result }: { result: ClaimQuestionResponse }) {
  const cited = new Set(result.cited_source_ids);

  return (
    <section className="result-panel claim-answer" aria-live="polite">
      <div className="result-heading">
        <p className="eyebrow">Claim question</p>
        <span className={`status-pill status-${result.status}`}>
          {statusLabels[result.status]}
        </span>
      </div>
      <div className="claim-answer-copy">
        {result.status === "answered" && (
          <p className="claim-answer-byline">Generated answer</p>
        )}
        <p className={result.status === "answered" ? "claim-answer-text" : "claim-answer-message"}>
          {result.answer || result.message}
        </p>
        {result.status === "answered" && (
          <p className="claim-answer-message">{result.message}</p>
        )}
      </div>
      {result.sources.length > 0 && (
        <div className="claim-answer-sources">
          <h3>{result.status === "answered" ? "Sources cited" : "Retrieved sources"}</h3>
          <ol>
            {result.sources.map((source) => (
              <li key={source.source_id}>
                <a
                  href={source.url}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={`${source.source_id}: ${source.title || source.source_name || source.url}`}
                >
                  <span className="source-number">{source.source_id}</span>
                  <span className="source-copy">
                    <strong>{source.title || source.source_name || "Retrieved source"}</strong>
                    <small>
                      {source.source_name || new URL(source.url).hostname}
                      {formatDate(source.published_at)
                        ? ` · Published ${formatDate(source.published_at)}`
                        : ""}
                      {cited.has(source.source_id) ? " · Cited in answer" : " · Retrieved"}
                    </small>
                  </span>
                  <span className="source-open" aria-hidden="true">↗</span>
                </a>
              </li>
            ))}
          </ol>
        </div>
      )}
      <p className="claim-answer-caution">{result.caution}</p>
    </section>
  );
}
