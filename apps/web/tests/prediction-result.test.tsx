import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PredictionResult } from "@/components/prediction-result";

describe("PredictionResult", () => {
  afterEach(() => vi.restoreAllMocks());

  it("renders repeated evidence URLs without duplicate React keys", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <PredictionResult
        result={{
          prediction: {
            available: true,
            label: "likely_real",
            confidence: 0.82,
            disclaimer: "A prediction is not proof.",
          },
          verification: {
            status: "supported",
            claims: [
              {
                claim_id: "claim-001",
                claim: "An example claim.",
                status: "supported",
                evidence: [
                  {
                    url: "https://example.org/article",
                    text: "First relevant passage.",
                    relevance_score: 0.8,
                    title: "Example article",
                  },
                  {
                    url: "https://example.org/article",
                    text: "Second relevant passage.",
                    relevance_score: 0.7,
                    title: "Example article",
                  },
                ],
              },
            ],
          },
        }}
      />,
    );

    expect(screen.getAllByRole("link", { name: /Example article/ })).toHaveLength(2);
    expect(consoleError).not.toHaveBeenCalledWith(
      expect.stringContaining("same key"),
      expect.anything(),
    );
  });

  it("clearly separates a transformer estimate from retrieved source passages", () => {
    render(
      <PredictionResult
        result={{
          prediction: {
            available: true,
            label: "likely_fake",
            confidence: 0.99,
            model: "transformer_sequence_classifier",
            disclaimer: "Text-pattern output is not a factual verdict.",
          },
          verification: {
            status: "sources_found",
            review_mode: "transformer_retrieval",
            claims: [
              {
                claim_id: "claim-001",
                claim: "NASA's Perseverance rover landed on Mars in 2021.",
                status: "sources_found",
                evidence: [
                  {
                    url: "https://science.nasa.gov/mars",
                    text: "The rover landed on Mars in 2021.",
                    relevance_score: 0.8,
                    title: "NASA Mars program",
                    source_classification: "primary_official",
                  },
                ],
              },
            ],
          },
        }}
      />,
    );

    expect(screen.getAllByText("Related sources found")).not.toHaveLength(0);
    expect(screen.getByRole("status")).toHaveTextContent(/Different signals/);
    expect(screen.getByText(/Official primary source/)).toBeInTheDocument();
    expect(screen.getByText("Writing-pattern estimate")).toBeInTheDocument();
    expect(screen.getByText("Uncalibrated model score")).toBeInTheDocument();
    expect(screen.getByText(/does not determine whether the claim is true/i)).toBeInTheDocument();
    expect(screen.queryByText("Evidence found")).not.toBeInTheDocument();

    const [sourceReview] = screen.getAllByText("Retrieved-source review");
    const styleEstimate = screen.getByText("Writing-pattern estimate");
    expect(
      sourceReview.compareDocumentPosition(styleEstimate) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});
