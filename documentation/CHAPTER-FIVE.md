# CHAPTER FIVE

# SUMMARY, CONCLUSION AND RECOMMENDATIONS

## 5.1 Summary of Findings

This study designed and implemented a full-stack system for analysing news article text through two separate processes: machine-learning classification and current-source evidence review. The work addressed the preparation of the ISOT Fake News Dataset, comparison of a classical text classifier with a transformer, retrieval of evidence for extracted claims, and presentation of the results through a web application and API. The system is intended to support closer reading and human review; it does not determine truth conclusively.

The data preparation pipeline cleaned and validated the ISOT records, removed empty and duplicate texts, excluded records with conflicting labels, and produced a reproducible stratified dataset. The resulting dataset contained 39,100 articles: 17,905 labelled `likely_fake` and 21,195 labelled `likely_real`. The split comprised 31,280 training records, 3,911 validation records, and 3,909 held-out test records. A separate temporal evaluation was also conducted for the classical model.

The TF-IDF and Logistic Regression model achieved 0.9900 accuracy and 0.9899 macro F1-score on the random held-out test set. It made 39 errors among 3,909 records. On the separate temporal test set, it achieved 0.9936 accuracy and 0.9839 macro F1-score. The fine-tuned DistilBERT model achieved 0.9995 accuracy and 0.9995 macro F1-score on the same random test set, with two errors. These results show that DistilBERT performed better on this benchmark split. They do not establish that it will perform better on unrelated publishers, later events, or Nigerian news. The transformer also produced very high confidence on its two incorrect predictions, demonstrating why confidence must not be interpreted as proof of accuracy.

The implementation provides a FastAPI endpoint that validates article text and returns a prediction separately from the evidence assessment. The API can select the classical or transformer predictor through configuration. Its evidence pipeline extracts a bounded number of checkable claims, uses a Brave Search adapter to obtain public web results when configured, fetches and ranks source material, extracts relevant passages, and reports `supported`, `contradicted`, `mixed`, or `insufficient` evidence. The Next.js client allows users to submit text and view the prediction, confidence, evidence, source links, and responsible-use information. Tests cover key model, retrieval, API, and frontend behaviour. The application has also been configured for a Render API and Vercel web deployment; free hosting is suitable for demonstration but does not establish service capacity or production reliability.

## 5.2 Conclusions

The study demonstrates that classical and transformer models can be trained and evaluated on a shared article-classification dataset and integrated into a working web application. On the random ISOT test split, DistilBERT obtained higher classification scores than TF-IDF with Logistic Regression. The classical model nevertheless remains useful as a lower-resource baseline, and its separate temporal evaluation provides an additional, though still dataset-limited, view of generalisation.

The study also demonstrates how text classification can be presented alongside current-source evidence without treating the two outputs as equivalent. The classifier estimates similarity to patterns learned from historical labelled articles. The retrieval component gathers candidate evidence for selected claims and may return insufficient evidence when sources are unavailable or inadequate. This separation helps communicate uncertainty and gives users source material to examine.

The project objectives were addressed through data preparation, model development and comparison, evidence retrieval, API and interface implementation, and automated testing. However, the results support the conclusion that the implemented system is a research and decision-support prototype, not an automated fact-checker. Its outputs require independent review, particularly when the subject is consequential or the available evidence is limited. The high scores on ISOT should be interpreted within the limits of that dataset and should not be generalised to all publishers, topics, languages, or current events.

## 5.3 Recommendations

- Evaluate the classifiers on recent news from publishers outside the ISOT dataset, including Nigerian sources, to test performance beyond the training data.
- Use publisher-based and time-based test splits to check whether the models rely on writing style or dataset-specific patterns.
- Review the DistilBERT model's errors and confidence scores on held-out examples before interpreting its predictions.
- Compare retrieved passages with human judgements to assess whether they are relevant to each claim; retain an “insufficient evidence” outcome when they are not.
- Present the classifier estimate and retrieved-source assessment as separate results, and remind users that neither is conclusive proof of truth.

## 5.4 Contribution to Knowledge

This study contributes a reproducible comparison of TF-IDF with Logistic Regression and fine-tuned DistilBERT for binary article classification using a common prepared ISOT dataset and held-out test set. Reporting class-sensitive metrics, confusion matrices, calibration measures, error observations, and a temporal evaluation of the classical model extends the comparison beyond a single headline accuracy score. The results also document the limits of interpreting near-perfect benchmark performance as evidence of real-world factual reliability.

The study further contributes an integrated prototype that presents learned text classification and current-source claim review as distinct outputs in one workflow. The implementation connects data and model artifacts, configurable inference, claim extraction, Brave Search retrieval, evidence passage ranking, a FastAPI service, and a responsive Next.js interface. This implementation provides a concrete basis for further research into evidence-aware news analysis, while its explicit insufficient-evidence state and source presentation show how uncertainty can remain visible to users.

The contribution is therefore both empirical and practical: an experimental comparison under shared dataset conditions, and a tested software architecture that exposes a model estimate together with retrievable evidence. It does not establish a validated method for determining truth, nor does it establish that either model is locally valid for contemporary Nigerian journalism.

## 5.5 Future Work

- **Nigerian news dataset:** Build and evaluate a labelled dataset covering local publishers, topics, and languages.
- **Cross-publisher model evaluation:** Study how fake-news classifiers perform across different publishers and time periods.
- **Semantic evidence verification:** Develop a system that classifies retrieved passages as supporting, contradicting, or unrelated to a claim.
- **Multimodal misinformation detection:** Extend detection to claims that combine text with images or video.
- **Conversational news research assistant:** Develop a question-answering interface that responds using retrieved articles and displays its sources.
- **User-centred evaluation:** Study how readers, journalists, and fact-checkers interpret model estimates and source evidence.
