export type EvidenceStatus =
  | "supported"
  | "contradicted"
  | "mixed"
  | "insufficient"
  | "sources_found";

export interface Prediction {
  available: boolean;
  label?: "likely_real" | "likely_fake" | null;
  confidence?: number | null;
  confidence_method?: "raw_softmax" | "temperature_scaled" | null;
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
  source_classification?: "primary_official" | "unclassified";
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
    review_mode?: "standard" | "transformer_retrieval";
  };
  explanation?: {
    available: boolean;
    text?: string | null;
    model?: string | null;
    error?: string | null;
  };
}

export interface ClaimQuestionSource {
  source_id: string;
  title: string;
  url: string;
  excerpt: string;
  relevance_score: number;
  source_name?: string | null;
  published_at?: string | null;
  retrieved_at?: string | null;
}

export interface ClaimQuestionResponse {
  status: "answered" | "insufficient_evidence" | "answer_unavailable";
  answer?: string | null;
  answer_engine?: string | null;
  cited_source_ids: string[];
  sources: ClaimQuestionSource[];
  message: string;
  caution: string;
}
