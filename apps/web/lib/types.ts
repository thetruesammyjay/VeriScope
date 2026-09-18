export type EvidenceStatus =
  | "supported"
  | "contradicted"
  | "mixed"
  | "insufficient";

export interface Prediction {
  available: boolean;
  label?: "likely_real" | "likely_fake" | null;
  confidence?: number | null;
  model?: string | null;
  model_version?: string | null;
  processing_time_ms?: number | null;
  error?: string | null;
  disclaimer: string;
}

export interface EvidencePassage {
  url: string;
  text: string;
  relevance_score: number;
  title?: string | null;
  source_name?: string | null;
  published_at?: string | null;
  retrieved_at?: string | null;
}

export interface ClaimAssessment {
  claim_id: string;
  claim: string;
  status: EvidenceStatus;
  rationale?: string | null;
  evidence: EvidencePassage[];
}

export interface AnalysisResponse {
  prediction: Prediction;
  verification: {
    status: EvidenceStatus;
    claims: ClaimAssessment[];
  };
}
