import Image from "next/image";
import { AnalysisForm } from "@/components/analysis-form";
import { SiteHeader } from "@/components/site-header";

export default function HomePage() {
  return (
    <main id="top">
      <SiteHeader />
      <section className="hero">
        <div className="hero-inner">
          <p className="eyebrow">Evidence-aware news analysis</p>
          <h1>Read beyond the headline.</h1>
          <p className="hero-copy">Use VeriScope to examine how an article reads, then review the current-source evidence behind its most checkable claims.</p>
          <a className="hero-link" href="#analyse">Start an analysis <span aria-hidden="true">↓</span></a>
        </div>
        <div className="hero-mascot">
          <Image
            src="/VeriScope-mascot.png"
            alt="VeriScope mascot holding a magnifying glass and verified document"
            width={400}
            height={400}
            priority
          />
        </div>
      </section>

      <section className="analysis-section" id="analyse">
        <div className="section-intro">
          <p className="eyebrow">Put it under the scope</p>
          <h2>One article. Two useful signals.</h2>
          <p>VeriScope keeps pattern classification and evidence review separate, so a model estimate is never presented as a final verdict.</p>
        </div>
        <AnalysisForm />
      </section>

      <section className="how-section" id="how-it-works">
        <div className="section-intro">
          <p className="eyebrow">How it works</p>
          <h2>Designed for a more careful read.</h2>
        </div>
        <div className="steps-grid">
          <article><span>01</span><h3>Read the language</h3><p>A trained model estimates whether the writing resembles patterns in its labelled news dataset.</p></article>
          <article><span>02</span><h3>Find claims</h3><p>VeriScope identifies a focused set of factual statements that can be checked externally.</p></article>
          <article><span>03</span><h3>Show the evidence</h3><p>Relevant source links and evidence states remain distinct from the model’s text estimate.</p></article>
        </div>
      </section>

      <section className="responsible-section" id="responsible-use">
        <div>
          <p className="eyebrow">Responsible use</p>
          <h2>A useful signal is not proof.</h2>
        </div>
        <p>VeriScope is a decision-support tool for research and closer reading. It can be wrong, especially with new events, unfamiliar publishers, satire, and content outside its training data. Verify important claims with credible independent sources.</p>
      </section>

      <footer>
        <span>VeriScope</span>
        <p>Evidence-aware analysis for more considered reading.</p>
      </footer>
    </main>
  );
}
