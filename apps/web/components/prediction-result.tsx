import type { AnalysisResponse, ClaimAssessment, EvidenceStatus } from "@/lib/types";
import { ConfidenceDisplay } from "./confidence-display";
import { Disclaimer } from "./disclaimer";

const statusCopy: Record<EvidenceStatus, string> = {
  supported: "Evidence found",
  contradicted: "Contradicting evidence",
  mixed: "Mixed evidence",
  insufficient: "Not enough evidence",
  sources_found: "Related sources found",
};

function reviewStatusCopy(status: EvidenceStatus, transformerReview: boolean) {
  if (!transformerReview) return statusCopy[status];
  if (status === "supported") return "Related passages found";
  if (status === "contradicted") return "Conflicting passages found";
  if (status === "mixed") return "Mixed source passages";
  return statusCopy[status];
}

function formatDate(date?: string | null) {
  if (!date) return null;
  const parsed = new Date(date);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toLocaleDateString();
}

function EvidenceReview({
  claims,
  status,
  transformerReview,
}: {
  claims: ClaimAssessment[];
  status: EvidenceStatus;
  transformerReview: boolean;
}) {
  return (
    <div className="evidence-section">
      <div>
        <p className="eyebrow">
          {transformerReview ? "Retrieved-source review" : "Current-source review"}
        </p>
        <h3>{reviewStatusCopy(status, transformerReview)}</h3>
        {transformerReview && (
          <p className="review-caution">
            Retrieved passages have lexical overlap with the claim. This does not
            determine whether the claim is true.
          </p>
        )}
      </div>
      {claims.length ? (
        <div className="claims-list">
          {claims.map((claim) => (
            <article className="claim" key={claim.claim_id}>
              <div className="claim-topline">
                <span className={`claim-status status-${claim.status}`}>
                  {reviewStatusCopy(claim.status, transformerReview)}
                </span>
                <span>
                  {claim.evidence.length} source
                  {claim.evidence.length === 1 ? "" : "s"}
                </span>
              </div>
              <p className="claim-text">“{claim.claim}”</p>
              {claim.rationale && (
                <p className="claim-rationale">{claim.rationale}</p>
              )}
              {claim.evidence.map((evidence, evidenceIndex) => (
                <a
                  className="evidence-link"
                  href={evidence.url}
                  key={`${claim.claim_id}-${evidence.url}-${evidenceIndex}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <span>{evidence.title || evidence.source_name || "View source"}</span>
                  <small>
                    {evidence.source_classification === "primary_official"
                      ? "Official primary source"
                      : "Other source"}
                    {formatDate(evidence.published_at)
                      ? ` · ${formatDate(evidence.published_at)}`
                      : ""} ↗
                  </small>
                </a>
              ))}
            </article>
          ))}
        </div>
      ) : (
        <p className="empty-evidence">
          No checkable claims or adequate current evidence were found. This is not
          confirmation that the article is accurate.
        </p>
      )}
    </div>
  );
}

export function PredictionResult({ result }: { result: AnalysisResponse }) {
  const { prediction, verification } = result;
  const label = prediction.label === "likely_real" ? "Likely real" : "Likely fake";
  const transformerReview = verification.review_mode === "transformer_retrieval";
  const signalsDiffer =
    transformerReview &&
    prediction.available &&
    prediction.label === "likely_fake" &&
    verification.status === "sources_found";

  const predictionCard = prediction.available ? (
    <div className={`prediction-card${transformerReview ? " prediction-card-secondary" : ""}`}>
      <p className="prediction-label">
        {transformerReview ? "Writing-pattern estimate" : "Text pattern estimate"}
      </p>
      <h2>{label}</h2>
      {prediction.confidence !== undefined && prediction.confidence !== null && (
        <ConfidenceDisplay
          confidence={prediction.confidence}
          label={
            transformerReview
              ? prediction.confidence_method === "temperature_scaled"
                ? "Temperature-scaled model score"
                : "Uncalibrated model score"
              : "Model confidence"
          }
        />
      )}
      <p className="model-note">
        {prediction.model?.replace(/_/g, " ")} {prediction.model_version ? `· ${prediction.model_version}` : ""}
      </p>
      {transformerReview && (
        <p className="prediction-caution">
          This text classifier estimates patterns learned from its training data;
          it does not establish factual accuracy.
        </p>
      )}
      {signalsDiffer && (
        <p className="signal-note" role="status">
          Different signals: the writing-pattern estimate and source review
          differ. The retrieved sources may be useful to inspect, but they do
          not independently verify this article.
        </p>
      )}
    </div>
  ) : (
    <div className="unavailable-card">
      <h2>Prediction unavailable</h2>
      <p>{prediction.error || "The trained model is not available right now."}</p>
    </div>
  );

  const evidenceReview = (
    <EvidenceReview
      claims={verification.claims}
      status={verification.status}
      transformerReview={transformerReview}
    />
  );

  return (
    <section className="result-panel" aria-live="polite">
      <div className="result-heading">
        <p className="eyebrow">
          {transformerReview ? "Retrieved-source review" : "Analysis result"}
        </p>
        <span className={`status-pill status-${verification.status}`}>
          {reviewStatusCopy(verification.status, transformerReview)}
        </span>
      </div>

      {transformerReview ? (
        <>
          {evidenceReview}
          {predictionCard}
        </>
      ) : (
        <>
          {predictionCard}
          {evidenceReview}
        </>
      )}

      <Disclaimer text={prediction.disclaimer} />
    </section>
  );
}
