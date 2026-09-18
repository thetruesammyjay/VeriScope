import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({ analyzeArticle: vi.fn() }));

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
});
