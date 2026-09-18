"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { analyzeArticle } from "@/lib/api";
import { maximumArticleLength, minimumArticleLength } from "@/lib/article-limits";
import type { AnalysisResponse } from "@/lib/types";
import { PredictionResult } from "./prediction-result";

export function AnalysisForm() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const article = text.trim();
    if (article.length < minimumArticleLength) {
      setError(`Add at least ${minimumArticleLength} characters so the analysis has enough context.`);
      setResult(null);
      return;
    }

    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setError(null);
    setIsLoading(true);
    try {
      setResult(await analyzeArticle(article, controller.signal));
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setResult(null);
      setError(cause instanceof Error ? cause.message : "Analysis could not be completed. Try again.");
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setIsLoading(false);
      }
    }
  }

  return (
    <div className="analysis-workspace">
      <form className="analysis-form" onSubmit={handleSubmit} noValidate>
        <div className="form-heading">
          <p className="eyebrow">Article workspace</p>
          <span>{text.trim().length.toLocaleString()} characters</span>
        </div>
        <label htmlFor="article-text">Paste the article text</label>
        <textarea
          id="article-text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Paste a news article or a substantial excerpt here…"
          maxLength={maximumArticleLength}
          aria-describedby="article-help"
        />
        <p id="article-help">VeriScope analyses text patterns and checks selected factual claims against currently available sources.</p>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="primary-button" type="submit" disabled={isLoading}>
          {isLoading ? "Analysing article…" : "Analyse article"} <span aria-hidden="true">↗</span>
        </button>
      </form>
      {isLoading && <div className="loading-panel" aria-live="polite"><span className="loading-mark" />Checking text patterns and current sources…</div>}
      {result && <PredictionResult result={result} />}
    </div>
  );
}
