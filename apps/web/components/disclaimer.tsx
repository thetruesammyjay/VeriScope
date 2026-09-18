export function Disclaimer({ text }: { text?: string }) {
  return (
    <p className="disclaimer">
      <span aria-hidden="true">i</span>
      {text || "VeriScope provides a model estimate, not a factual verdict. Check important claims with reliable primary or professional fact-checking sources."}
    </p>
  );
}
