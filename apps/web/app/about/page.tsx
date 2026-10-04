import type { Metadata } from "next";
import { SiteHeader } from "@/components/site-header";

export const metadata: Metadata = {
  title: "About | VeriScope",
  description:
    "Learn what VeriScope does, how its analysis tools work, and where their limits are.",
};

export default function AboutPage() {
  return (
    <main>
      <SiteHeader />
      <section className="info-hero">
        <div className="info-hero-inner">
          <p className="eyebrow">About VeriScope</p>
          <h1>More context.<br />Less certainty theater.</h1>
          <p className="info-lede">
            VeriScope is an academic research project and news-analysis tool. It
            brings together a learned writing-pattern estimate and links to
            current sources, while keeping those signals separate.
          </p>
          <a className="primary-button info-action" href="/#analyse">
            Try an analysis <span aria-hidden="true">↗</span>
          </a>
        </div>
        <div className="info-hero-mark" aria-hidden="true">
          <span>TEXT</span><i /><span>SOURCES</span>
        </div>
      </section>

      <section className="info-section">
        <div className="info-section-heading">
          <p className="eyebrow">What it does</p>
          <h2>Two ways to examine a news claim.</h2>
        </div>
        <div className="info-cards">
          <article className="info-card info-card-paper">
            <span className="info-index">01 / ARTICLE ANALYSIS</span>
            <h3>Inspect the text and its claims.</h3>
            <p>
              The article workflow returns a classifier estimate based on
              learned language patterns. Separately, it extracts checkable
              claims and searches for related passages and source pages.
            </p>
            <p className="info-card-note">
              A writing-pattern estimate is not a finding about whether an
              article is true.
            </p>
          </article>
          <article className="info-card info-card-dark">
            <span className="info-index">02 / CLAIM QUESTION</span>
            <h3>Ask a focused question.</h3>
            <p>
              The question workflow searches the web, selects relevant passages,
              and can generate a concise answer grounded in those retrieved
              excerpts. Source links are provided so readers can inspect them.
            </p>
            <p className="info-card-note">
              This generated response is a separate demo feature, not the
              DistilBERT research classifier.
            </p>
          </article>
        </div>
      </section>

      <section className="info-section info-section-bordered">
        <div className="info-section-heading">
          <p className="eyebrow">Read the result carefully</p>
          <h2>What VeriScope cannot establish.</h2>
        </div>
        <div className="info-boundaries">
          <p>
            Finding a page that discusses a claim does not by itself confirm
            that claim. Search coverage is incomplete, pages can be outdated or
            unreliable, and lexical similarity can miss context, nuance, or
            contradiction.
          </p>
          <p>
            The classifier can also make confident mistakes, especially on
            short text, unfamiliar topics, and writing unlike its training
            examples. Its confidence describes the model output; it is not the
            probability that a claim is true.
          </p>
        </div>
        <a className="text-link" href="/methodology">
          See the methodology <span aria-hidden="true">↗</span>
        </a>
      </section>

      <footer>
        <span>VeriScope</span>
        <p>Evidence-aware analysis for more considered reading.</p>
      </footer>
    </main>
  );
}
