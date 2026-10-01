export function ConfidenceDisplay({
  confidence,
  label = "Model confidence",
}: {
  confidence: number;
  label?: string;
}) {
  const percentage = Math.round(confidence * 100);

  return (
    <div className="confidence" aria-label={`${percentage}% ${label.toLowerCase()}`}>
      <div className="confidence-copy">
        <span>{label}</span>
        <strong>{percentage}%</strong>
      </div>
      <div className="confidence-track" aria-hidden="true">
        <span style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}
