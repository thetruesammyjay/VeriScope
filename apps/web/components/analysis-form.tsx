"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { analyzeArticle, askAboutClaim } from "@/lib/api";
import { maximumArticleLength, minimumArticleLength } from "@/lib/article-limits";
import type { AnalysisResponse, ClaimQuestionResponse } from "@/lib/types";
import { ClaimAnswer } from "./claim-answer";
import { PredictionResult } from "./prediction-result";

type Workflow = "article" | "question";
const minimumQuestionLength = 5;
const maximumQuestionLength = 500;

export function AnalysisForm() {
  const [workflow, setWorkflow] = useState<Workflow>("article");
  const [articleText, setArticleText] = useState("");
  const [questionText, setQuestionText] = useState("");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [questionResult, setQuestionResult] = useState<ClaimQuestionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  function changeWorkflow(next: Workflow) {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setWorkflow(next);
    setError(null);
    setResult(null);
    setQuestionResult(null);
    setIsLoading(false);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const input = (isArticle ? articleText : questionText).trim();
    if (workflow === "article" && input.length < minimumArticleLength) {
      setError(`Add at least ${minimumArticleLength} characters so the analysis has enough context.`);
      setResult(null);
      setQuestionResult(null);
      return;
    }
    if (workflow === "question" && input.length < minimumQuestionLength) {
      setError(`Enter a question with at least ${minimumQuestionLength} characters.`);
      setResult(null);
      setQuestionResult(null);
      return;
    }

    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setError(null);
    setIsLoading(true);
    try {
      if (workflow === "question") {
        setQuestionResult(await askAboutClaim(input, controller.signal));
        setResult(null);
      } else {
        setResult(await analyzeArticle(input, controller.signal));
        setQuestionResult(null);
      }
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setResult(null);
      setQuestionResult(null);
      setError(cause instanceof Error ? cause.message : "Analysis could not be completed. Try again.");
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setIsLoading(false);
      }
    }
  }

  const isArticle = workflow === "article";
  const text = isArticle ? articleText : questionText;
  return (
    <div className="analysis-workspace">
      <form className="analysis-form" onSubmit={handleSubmit} noValidate>
        <div className="workflow-toggle" role="group" aria-label="Choose a workflow">
          <button type="button" aria-pressed={isArticle} className={isArticle ? "is-selected" : ""} onClick={() => changeWorkflow("article")}>
            Article analysis
          </button>
          <button type="button" aria-pressed={!isArticle} className={!isArticle ? "is-selected" : ""} onClick={() => changeWorkflow("question")}>
            Ask about a claim
          </button>
        </div>
        <div className="form-heading">
          <p className="eyebrow">{isArticle ? "Article workspace" : "Claim question"}</p>
          <span>{text.trim().length.toLocaleString()} characters</span>
        </div>
        <label htmlFor="article-text">
          {isArticle ? "Paste the article text" : "What news claim do you want to check?"}
        </label>
        <textarea
          id="article-text"
          value={text}
          onChange={(event) => {
            if (isArticle) setArticleText(event.target.value);
            else setQuestionText(event.target.value);
          }}
          placeholder={isArticle ? "Paste a news article or a substantial excerpt here…" : "For example: Did Muhammadu Buhari die?"}
          maxLength={isArticle ? maximumArticleLength : maximumQuestionLength}
          aria-describedby="article-help"
        />
        <p id="article-help">
          {isArticle
            ? "VeriScope analyses text patterns and checks selected factual claims against currently available sources."
            : "Ask about a news claim. VeriScope searches current sources and generates a cautious answer from retrieved passages."}
        </p>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="primary-button" type="submit" disabled={isLoading}>
          {isLoading ? (isArticle ? "Analysing article…" : "Searching sources…") : (isArticle ? "Analyse article" : "Search and answer")}
          <span aria-hidden="true">↗</span>
        </button>
      </form>
      {isLoading && (
        <div className="loading-panel" aria-live="polite">
          <span className="loading-mark" />
          {isArticle ? "Checking text patterns and current sources…" : "Searching current sources and preparing an answer…"}
        </div>
      )}
      {result && <PredictionResult result={result} />}
      {questionResult && <ClaimAnswer result={questionResult} />}
    </div>
  );
}
