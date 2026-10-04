import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({ analyzeArticle: vi.fn(), askAboutClaim: vi.fn() }));

vi.mock("@/lib/api", () => apiMocks);

import { AnalysisForm } from "@/components/analysis-form";

describe("AnalysisForm", () => {
  it("explains the minimum-context requirement before sending a request", () => {
    render(<AnalysisForm />);

    fireEvent.click(screen.getByRole("button", { name: /analyse article/i }));

    expect(
      screen.getByText(/Add at least 100 characters so the analysis has enough context/i),
    ).toBeInTheDocument();
    expect(apiMocks.analyzeArticle).not.toHaveBeenCalled();
  });

  it("submits an article and renders the returned model result", async () => {
    apiMocks.analyzeArticle.mockResolvedValueOnce({
      prediction: {
        available: true,
        label: "likely_real",
        confidence: 0.82,
        model: "tfidf_logistic_regression",
        model_version: "test",
        disclaimer: "A prediction is not proof.",
      },
      verification: { status: "insufficient", claims: [] },
    });
    render(<AnalysisForm />);
    const article = "a".repeat(100);

    fireEvent.change(screen.getByLabelText("Paste the article text"), {
      target: { value: article },
    });
    fireEvent.click(screen.getByRole("button", { name: /analyse article/i }));

    await waitFor(() => {
      expect(apiMocks.analyzeArticle).toHaveBeenCalledWith(article, expect.any(AbortSignal));
    });
    expect(await screen.findByText("Likely real")).toBeInTheDocument();
    expect(screen.getByText("82%")).toBeInTheDocument();
  });

  it("uses the separate claim-question endpoint without running article classification", async () => {
    apiMocks.askAboutClaim.mockResolvedValueOnce({
      status: "answered",
      answer: "Yes, the retrieved source reports that he died. [S1]",
      answer_engine: "DeepSeek",
      cited_source_ids: ["S1"],
      sources: [{
        source_id: "S1",
        title: "Official announcement",
        url: "https://example.org/announcement",
        excerpt: "A retrieved passage.",
        relevance_score: 1,
      }],
      message: "Answer generated from retrieved passages.",
      caution: "AI-generated answer; review the linked source.",
    });
    render(<AnalysisForm />);
    fireEvent.click(screen.getByRole("button", { name: "Ask about a claim" }));
    fireEvent.change(screen.getByLabelText("What news claim do you want to check?"), {
      target: { value: "Did this happen?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /search and answer/i }));

    await waitFor(() => {
      expect(apiMocks.askAboutClaim).toHaveBeenCalledWith("Did this happen?", expect.any(AbortSignal));
    });
    expect(await screen.findByText("Yes, the retrieved source reports that he died. [S1]")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "S1: Official announcement" })).toHaveAttribute(
      "href",
      "https://example.org/announcement",
    );
    expect(apiMocks.analyzeArticle).not.toHaveBeenCalled();
  });
});
