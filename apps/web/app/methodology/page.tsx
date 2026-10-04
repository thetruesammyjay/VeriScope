import type { Metadata } from "next";
import { SiteHeader } from "@/components/site-header";

export const metadata: Metadata = {
  title: "Methodology | VeriScope",
  description:
    "Explore the research methodology, model development, evaluation, and evidence workflows behind VeriScope.",
};

const stages = [
  {
    number: "01",
    title: "Prepare the data",
    text: "The classifier study uses the ISOT Fake and Real News dataset. The records are cleaned and deduplicated, then divided into stratified training, validation, and test sets.",
  },
  {
    number: "02",
    title: "Train and compare",
    text: "A TF-IDF with Logistic Regression baseline is compared with a fine-tuned DistilBERT sequence classifier. Both learn labels from the dataset; neither directly checks a story against reality.",
  },
  {
    number: "03",
    title: "Evaluate separately",
    text: "Held-out data is used to assess classification performance and inspect errors. The source-retrieval workflow is a separate component and should be evaluated for relevance and coverage, not treated as classifier evidence.",
  },
  {
    number: "04",
    title: "Integrate the workflows",
    text: "The API coordinates article analysis and returns distinct model and source-review results. The interface presents them separately to avoid implying that one signal verifies the other.",
  },
];

export default function MethodologyPage() {
  return (
    <main>
      <SiteHeader />
      <section className="info-hero methodology-hero">
        <div className="info-hero-inner">
          <p className="eyebrow">Methodology</p>
          <h1>Build, test,<br />then interpret.</h1>
          <p className="info-lede">
            The project combines an iterative software process with a machine
            learning development lifecycle. Its research classifier and live
            source workflows have different jobs and are described separately.
          </p>
        </div>
        <div className="method-stamp" aria-label="Research process: Scrum and machine learning lifecycle">
          <span>SCRUM</span><b>+</b><span>MLDLC</span>
        </div>
      </section>

      <section className="info-section">
        <div className="info-section-heading">
          <p className="eyebrow">Research process</p>
          <h2>From labelled examples to a working system.</h2>
          <p>
            Scrum supports incremental planning and implementation; the Machine
            Learning Development Life Cycle (MLDLC) guides data preparation,
            model development, evaluation, and integration.
          </p>
        </div>
        <ol className="method-stages">
          {stages.map((stage) => (
            <li key={stage.number}>
              <span className="method-number">{stage.number}</span>
              <div>
                <h3>{stage.title}</h3>
                <p>{stage.text}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="info-section methodology-signals">
        <div className="info-section-heading">
          <p className="eyebrow">Two distinct workflows</p>
          <h2>Do not confuse prediction with evidence.</h2>
        </div>
        <div className="signal-grid">
          <article>
            <span className="signal-tag">RESEARCH CLASSIFIER</span>
            <h3>TF-IDF baseline / DistilBERT</h3>
            <p>
              The article classifier estimates a label from patterns learned in
              labelled news text. The model score is not a factuality score and
              should be interpreted with held-out evaluation results and known
              dataset limitations.
            </p>
          </article>
          <article>
            <span className="signal-tag">LIVE DEMO WORKFLOW</span>
            <h3>Brave Search / retrieved passages</h3>
            <p>
              Search results are deduplicated and filtered, pages are fetched
              where possible, and passages are selected by lexical relevance.
              For focused questions, DeepSeek can generate an answer using the
              retrieved excerpts. This is not a trained classifier or a
              guarantee of truth.
            </p>
          </article>
        </div>
      </section>

      <section className="info-section info-section-bordered method-limitations">
        <div className="info-section-heading">
          <p className="eyebrow">Interpretation and limits</p>
          <h2>Results depend on data and sources.</h2>
        </div>
        <p>
          Performance on the study dataset may not generalize to new publishers,
          current events, Nigerian news, or other writing styles. Retrieved
          sources may be incomplete, irrelevant, or wrong. Review the original
          pages and use independent sources for consequential decisions.
        </p>
        <a className="text-link" href="/about">
          About VeriScope <span aria-hidden="true">↗</span>
        </a>
      </section>

      <footer>
        <span>VeriScope</span>
        <p>Evidence-aware analysis for more considered reading.</p>
      </footer>
    </main>
  );
}
