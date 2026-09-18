export function ConfidenceDisplay({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100);

  return (
    <div className="confidence" aria-label={`${percentage}% model confidence`}>
      <div className="confidence-copy">
        <span>Model confidence</span>
        <strong>{percentage}%</strong>
      </div>
      <div className="confidence-track" aria-hidden="true">
        <span style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}
