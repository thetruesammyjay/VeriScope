import type { AnalysisResponse, EvidenceStatus } from "@/lib/types";
import { ConfidenceDisplay } from "./confidence-display";
import { Disclaimer } from "./disclaimer";

const statusCopy: Record<EvidenceStatus, string> = {
  supported: "Evidence found",
  contradicted: "Contradicting evidence",
  mixed: "Mixed evidence",
  insufficient: "Not enough evidence",
};

function formatDate(date?: string | null) {
  if (!date) return null;
  const parsed = new Date(date);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toLocaleDateString();
}

export function PredictionResult({ result }: { result: AnalysisResponse }) {
  const { prediction, verification } = result;
  const label = prediction.label === "likely_real" ? "Likely real" : "Likely fake";

  return (
    <section className="result-panel" aria-live="polite">
      <div className="result-heading">
        <p className="eyebrow">Analysis result</p>
        <span className={`status-pill status-${verification.status}`}>{statusCopy[verification.status]}</span>
      </div>

      {prediction.available ? (
        <div className="prediction-card">
          <p className="prediction-label">Text pattern estimate</p>
          <h2>{label}</h2>
          {prediction.confidence !== undefined && prediction.confidence !== null && (
            <ConfidenceDisplay confidence={prediction.confidence} />
          )}
          <p className="model-note">{prediction.model?.replace(/_/g, " ")} {prediction.model_version ? `· ${prediction.model_version}` : ""}</p>
        </div>
      ) : (
        <div className="unavailable-card">
          <h2>Prediction unavailable</h2>
          <p>{prediction.error || "The trained model is not available right now."}</p>
        </div>
      )}

      <div className="evidence-section">
        <div>
          <p className="eyebrow">Current-source review</p>
          <h3>{statusCopy[verification.status]}</h3>
        </div>
        {verification.claims.length ? (
          <div className="claims-list">
            {verification.claims.map((claim) => (
              <article className="claim" key={claim.claim_id}>
                <div className="claim-topline">
                  <span className={`claim-status status-${claim.status}`}>{statusCopy[claim.status]}</span>
                  <span>{claim.evidence.length} source{claim.evidence.length === 1 ? "" : "s"}</span>
                </div>
                <p className="claim-text">“{claim.claim}”</p>
                {claim.rationale && <p className="claim-rationale">{claim.rationale}</p>}
                {claim.evidence.map((evidence) => (
                  <a className="evidence-link" href={evidence.url} key={`${claim.claim_id}-${evidence.url}`} target="_blank" rel="noreferrer">
                    <span>{evidence.title || evidence.source_name || "View source"}</span>
                    <small>{formatDate(evidence.published_at) || "Retrieved source"} ↗</small>
                  </a>
                ))}
              </article>
            ))}
          </div>
        ) : (
          <p className="empty-evidence">No checkable claims or adequate current evidence were found. This is not confirmation that the article is accurate.</p>
        )}
      </div>

      <Disclaimer text={prediction.disclaimer} />
    </section>
  );
}
