# CHAPTER ONE

# INTRODUCTION

## 1.1 Background to the Study

For many people, the first account of an important event now arrives through a phone rather than a newspaper or scheduled broadcast. News websites, social networks, and messaging applications can carry a report to thousands of readers within minutes. This speed has made information easier to access, but it has also made false or misleading stories easier to circulate. By the time journalists or professional fact-checkers investigate a questionable report, it may already have been copied, discussed, and shared across several platforms.

This problem is especially relevant in Nigeria, where online reports shape conversations about politics, public health, security, business, and community life. A false story can move from a social-media post to a messaging group and then reappear on another platform without its original context. During elections, disease outbreaks, or security incidents, readers may face several conflicting accounts at once and may not have the evidence needed to judge them. Ahmed and Msughter (2022), in their study of COVID-19 information among social-media users in Kano State, linked exposure to fake news with consequences such as failure to follow safety measures. The concern, then, is not simply that false information exists. It is that it travels quickly and often looks convincing.

Although the expressions *misinformation*, *disinformation*, and *fake news* are often used interchangeably, they do not mean exactly the same thing. Misinformation is inaccurate information that may be shared without an intention to cause harm. Disinformation is deliberately created or circulated to deceive. Fake news usually refers to fabricated or seriously misleading content presented in the form of a genuine news report. For the purpose of this study, fake-news detection is framed as supervised text classification. The model reads the text of an article and estimates whether its language is more similar to articles labelled real or fake in the training data.

That definition sets an important boundary around what a text classifier can claim. On its own, it does not interview witnesses, consult official records, or check a report against current evidence. It learns patterns in vocabulary, style, topic, punctuation, and document structure. Some of those patterns may be useful, but none of them proves that a claim is true. For this reason, the proposed system adds an evidence-retrieval stage to the classifier. It identifies checkable claims in the article, searches for recent information from relevant and credible online sources, and compares the retrieved evidence with those claims. The system remains a decision-support tool, but its assessment is informed by both learned language patterns and evidence available at the time of analysis.

The two forms of analysis are kept visible rather than being treated as the same thing. The classification component produces the cautious labels **Likely Real** and **Likely Fake** with a confidence score. The evidence component reports whether the retrieved information supports, contradicts, is mixed about, or is insufficient to assess a claim. It also presents source links, publication dates, and the time of retrieval. This distinction reduces the risk that a strong linguistic prediction will be mistaken for factual proof or that weak search evidence will be presented as conclusive.

Natural Language Processing (NLP) makes it possible for a computer to represent and analyse written language. In this case, article text must first be converted into numerical features before a machine-learning model can work with it. Traditional approaches use features such as word frequency, n-grams, and term weights. Ahmed, Traore, and Saad (2017, 2018) showed that these features can help distinguish labelled fake articles from truthful ones. The approach is efficient and provides a useful baseline, although it captures observed word patterns more readily than the wider meaning of a sentence or document.

Term Frequency-Inverse Document Frequency (TF-IDF) is a widely used representation for classical text classification. It increases the importance of terms that occur frequently in a document but are less common across the entire document collection. A classifier such as Logistic Regression can then learn weights associated with terms or n-grams and produce a probability for each class. This combination is suitable as a baseline because it trains comparatively quickly, supports fast CPU inference, and makes it possible to inspect influential features.

Transformer models provide a more contextual approach to language representation. BERT introduced deep bidirectional pre-training that allows a model to interpret a word using the text that appears before and after it (Devlin et al., 2019). DistilBERT applies knowledge distillation to produce a smaller and faster version of BERT while retaining much of its language-understanding capability (Sanh et al., 2019). A fine-tuned DistilBERT classifier can therefore learn contextual patterns that a sparse TF-IDF representation may not capture. However, transformer training and inference generally require more computation, memory, and deployment resources than a classical model.

Rather than assume that the newer model will be better, this study compares both approaches under the same conditions. The classical branch combines TF-IDF with Logistic Regression, with a Linear Support Vector Machine as an optional benchmark. The second branch fine-tunes DistilBERT for the same binary task. Both models use the same reproducible data splits and evaluation procedure. Accuracy, precision, recall, F1-score, confusion matrices, error patterns, inference time, model size, and deployment needs are considered together. In this way, the final choice is based on evidence rather than model complexity alone.

The primary data source is the ISOT Fake News Dataset created by the Information Security and Object Technology Research Lab at the University of Victoria. It contains thousands of labelled truthful and fake news articles and is suited to an article-level workflow. The dataset is preferred to the LIAR benchmark for this study because LIAR consists mainly of short political statements, while the proposed system is designed to analyse fuller news-article text. The data pipeline loads the original `Fake.csv` and `True.csv` files, assigns labels, aligns fields, removes null and duplicate records, constructs the classification text, and generates reproducible stratified splits.

Careful preparation of the dataset is just as important as the choice of algorithm. If duplicate or nearly identical articles appear in both the training and test sets, the final score may look impressive even though the model has learned little that is useful. There is also a risk that the model will recognise a publisher or a recurring topic instead of learning patterns that transfer to other news sources. To make the experiment traceable, the study records its cleaning decisions, random seeds, dataset version, model settings, and outputs. False positives and false negatives are examined alongside the overall scores.

The implemented system combines the research pipeline with a usable full-stack application. A Next.js web application provides the user interface, while a FastAPI service validates requests and coordinates prediction and evidence retrieval. Shared Python modules handle data preparation, classical training, transformer fine-tuning, evaluation, artifact metadata, model loading, and production inference. When a user submits an article, the selected model first analyses its language. A separate verification workflow then extracts important factual claims, creates search queries, retrieves current material from selected online sources, ranks the evidence for relevance and source quality, and compares it with the claims. The response contains the model label and confidence, an evidence status, source citations, retrieval dates, model information, processing time, and a responsible-use disclaimer.

The architecture separates offline model development from online analysis. Training and evaluation occur outside the production request path, and only a validated artifact referenced by production metadata is loaded by the API. Current evidence, however, is retrieved when an article is analysed because it cannot be fixed at training time. Search results are filtered by relevance, date, accessibility, and source policy. The retrieved pages are treated as evidence candidates rather than automatically accepted as true. This separation allows the trained model to remain reproducible while the evidence layer can respond to recent events.

The work therefore goes beyond training a classifier in a notebook. It brings together data preparation, two modelling approaches, comparative evaluation, model management, a documented REST API, a responsive interface, testing, experiment records, and responsible-use guidance. At its centre are two related questions: how does a classical text classifier compare with a transformer on the same task, and how can the selected model be made available to users without presenting its output as unquestionable truth?

## 1.2 Statement of the Problem

Online stories can reach a large audience long before anyone has checked their claims, sources, or context. Some misleading reports are presented with the language and layout of professional journalism, making them difficult for an ordinary reader to recognise. Repetition can add an appearance of credibility: a story seen on several platforms may feel reliable even when each version comes from the same false source. Manual verification remains necessary, but it is much slower than sharing.

Although professional fact-checking remains essential, it is labour-intensive and cannot always provide an immediate first assessment for every reader. There is therefore a need to investigate automated tools that can analyse article text and identify patterns associated with labelled real and fake news. Such tools can support screening, research, media monitoring, and reader awareness, provided their limitations are communicated clearly.

There is also a practical question about which kind of model to use. A TF-IDF classifier is comparatively inexpensive to train and serve, but it may overlook context. A transformer can model context more effectively, yet it demands more memory, computation, and deployment effort. Evaluating only one of these approaches would leave that trade-off unanswered. They need to be compared on the same data and under the same evaluation conditions.

Even a high accuracy score can conceal serious weaknesses. If a credible article is wrongly labelled fake, the result may unfairly damage trust. If a fabricated article is labelled real, the output may give a reader false reassurance. Leakage, duplicated articles, source bias, class imbalance, and changes between the training data and real-world news can make the reported performance look better than it is. For that reason, this study considers class-level results, confusion matrices, typical errors, latency, and model size, not accuracy alone.

There is also an implementation gap between an experimental model and a responsible user-facing system. A notebook result does not provide input validation, stable model loading, current-source retrieval, evidence ranking, source citations, error handling, model versioning, responsive presentation, or disclaimers. Without these elements, users may misinterpret a score or assume that a linguistic classifier has checked the article against recent facts. A responsible system must also handle cases in which search results are unavailable, conflicting, outdated, or insufficient.

In response, this study designs and implements a full-stack automated fake-news detection system that prepares the ISOT dataset, trains a TF-IDF and Logistic Regression baseline, fine-tunes DistilBERT, and evaluates both approaches under a shared framework. The selected classifier is combined with claim extraction, current-source search, evidence ranking, and evidence-based verification. Predictions and supporting or contradicting sources are served through a web API and presented through a responsive interface using cautious labels and responsible-use information.

## 1.3 Objectives of the Study

The general objective of this study is to design and implement a full-stack system that uses NLP and machine-learning models to estimate whether a news article is likely real or likely fake from its textual content.

The specific objectives are to:

1. develop a reproducible pipeline for loading, validating, cleaning, labelling, and splitting the ISOT Fake News Dataset;
2. implement a classical fake-news classification baseline using TF-IDF and Logistic Regression, with an optional Linear SVM benchmark;
3. fine-tune a DistilBERT transformer model on the same binary article-classification task;
4. evaluate the candidate models using accuracy, precision, recall, F1-score, confusion matrices, class-wise results, inference latency, model size, and representative error analysis;
5. select and export a validated production model with versioned metadata and metrics;
6. implement a claim-extraction and search component that retrieves recent information from relevant and credible online sources;
7. develop an evidence-analysis component that ranks retrieved material and identifies whether it supports, contradicts, or provides insufficient information for the extracted claims;
8. develop a FastAPI service that returns the predicted class, confidence score, evidence status, cited sources, retrieval date, model identity, processing time, and disclaimer;
9. develop a responsive Next.js interface through which users can submit article text and review both the model prediction and the retrieved evidence; and
10. implement automated tests and documentation covering data processing, preprocessing, model utilities, search and retrieval behaviour, evidence handling, API responses, frontend states, system limitations, and responsible use.

## 1.4 Scope of the Study

This study covers the design and implementation of an English-language fake-news detection and evidence-retrieval system. The machine-learning scope includes ISOT dataset ingestion, schema validation, invalid-record and duplicate handling, text construction, label encoding, stratified train-validation-test splitting, classical preprocessing, TF-IDF feature extraction, Logistic Regression training, optional Linear SVM comparison, DistilBERT tokenisation and fine-tuning, shared evaluation, experiment reporting, model selection, and production artifact loading.

The evidence-retrieval scope includes extracting a limited set of checkable claims from an article, generating search queries, retrieving recent public web sources, recording URLs and retrieval times, filtering and ranking results, extracting relevant passages, and assigning an evidence status of supported, contradicted, mixed, or insufficient evidence. Preference is given to primary sources, official records, established news organisations, and professional fact-checking publications where appropriate. Search rank alone is not treated as proof of credibility.

The application scope includes a Next.js and TypeScript frontend, a FastAPI and Pydantic backend, request validation, environment-based configuration, model, search, evidence, and metrics endpoints, health checks, error handling, confidence presentation, cited evidence, and a responsible-use disclaimer. Automated testing covers central Python modules, API behaviour, retrieval failures, conflicting evidence, frontend components, and the main browser workflow.

The study uses the ISOT dataset as its primary classification benchmark and treats each input as article text. It does not build a general-purpose crawler, monitor private messages, or automatically collect user data. It does not perform image, audio, or video verification or analyse social-network propagation. Web retrieval is limited to publicly accessible textual sources returned for specific extracted claims. The system does not guarantee that every relevant source is indexed, accessible, current, independent, or correct, and it does not automatically remove, censor, or penalise content.

The system is not a substitute for professional fact-checking. Its classifier reflects patterns learned from historical labelled examples, while its retrieval component depends on the quality and availability of online sources. Newly emerging events may have little reliable evidence, and different credible sources may disagree. The classifier may also perform poorly on Nigerian local journalism, satire, scientific reporting, multilingual content, machine-generated articles, or other material outside the training distribution. The application therefore presents classification and evidence findings separately and allows an **Insufficient Evidence** outcome.

## 1.5 Significance of the Study

For readers, the proposed system provides more than a model score. It offers a preliminary linguistic assessment and shows recent evidence that may support or challenge important claims in the article. Source links, dates, and an insufficient-evidence option make uncertainty more visible and give readers a starting point for further verification.

For journalists, fact-checkers, moderators, and media-monitoring teams, the proposed system provides a prototype screening tool that may help prioritise content for human review and locate potentially relevant sources. It does not replace investigation, but it demonstrates how classification, search, and evidence comparison can contribute to a broader verification workflow.

For machine-learning researchers and students, the study offers a reproducible comparison between a strong classical baseline and a fine-tuned transformer. Using the same data split and evaluation utilities makes the performance-cost trade-off more defensible. The inclusion of confusion matrices, error analysis, latency, artifact size, and experiment metadata also encourages evaluation beyond a headline accuracy value.

For software engineering practice, the study demonstrates the path from an experimental NLP pipeline to a modular application. It connects Python model development, artifact versioning, FastAPI inference, Pydantic validation, Next.js presentation, automated testing, and environment-based configuration while maintaining a clear separation between training and serving.

For Nigerian academic and social contexts, the study contributes a practical discussion of automated misinformation detection while explicitly acknowledging domain limitations. It provides a foundation for future evaluation on locally sourced Nigerian news, multilingual material, source-aware features, claim-level evidence retrieval, and collaboration with professional fact-checkers.

Finally, the study contributes to responsible AI practice by avoiding absolute claims. It distinguishes statistical classification from factual verification, records known limitations, discourages automated censorship, and requires human judgement for consequential decisions.

## 1.6 Definition of Terms

**Accuracy:** The proportion of all evaluated examples that a model classifies correctly.

**Application Programming Interface (API):** A defined interface through which the frontend sends article text to the backend and receives prediction results.

**Artificial Intelligence (AI):** The broad field concerned with computer systems that perform tasks commonly associated with human intelligence, including language analysis and pattern recognition.

**Classification:** The process of assigning an input to one of a defined set of categories. In this study, the categories are likely real and likely fake.

**Claim Extraction:** The identification of factual statements in an article that can reasonably be checked against external evidence.

**Confidence Score:** A model-derived numerical value representing the strength of a prediction. It is not proof or factual certainty.

**Confusion Matrix:** A table that summarises correct and incorrect predictions for each class, including false positives and false negatives.

**DistilBERT:** A compact transformer language model produced by distilling knowledge from BERT and suitable for fine-tuning on text-classification tasks.

**Domain Shift:** A change between the data on which a model was trained and the data it encounters after deployment.

**Evidence Retrieval:** The process of searching for and selecting source passages relevant to an extracted claim.

**Evidence Status:** The system's summary of whether retrieved evidence supports, contradicts, is mixed about, or is insufficient to assess a claim.

**F1-Score:** The harmonic mean of precision and recall, used to balance both measures in classification evaluation.

**Fake News:** For this study, article-style content labelled as fabricated or misleading in the research dataset. The term does not imply that the system can independently establish truth.

**FastAPI:** A Python web framework used in the implementation to expose the model's inference service and documented endpoints.

**Fine-Tuning:** The process of adapting a pretrained model to a specific labelled task by continuing its training on task-specific data.

**Inference:** The use of a trained model to generate a prediction for previously unseen input.

**ISOT Fake News Dataset:** A labelled collection of truthful and fake news articles released by the University of Victoria's ISOT Research Lab and used as the primary dataset in this study.

**Logistic Regression:** The supervised statistical classifier used in this study as the primary classical baseline.

**Machine Learning:** A field of computing in which algorithms learn patterns from data to make predictions or decisions.

**Natural Language Processing (NLP):** Computational methods for representing, analysing, and generating human language.

**Next.js:** The React-based web framework used to implement the proposed system's user interface.

**Precision:** The proportion of examples predicted as belonging to a class that actually belong to that class in the labelled evaluation data.

**Recall:** The proportion of labelled examples in a class that the model correctly identifies.

**Retrieval-Augmented Verification:** A process in which information retrieved from external sources is supplied to an NLP component to support the assessment of a claim.

**Source Credibility:** The degree to which a source is considered authoritative, transparent, relevant, and dependable for the claim being examined.

**Temporal Relevance:** The extent to which retrieved evidence is recent enough, or appropriately dated, for the event or claim under examination.

**Stratified Split:** A division of data that preserves approximately the same class proportions across training, validation, and test sets.

**TF-IDF:** Term Frequency-Inverse Document Frequency, a numerical weighting method that represents how important a term is to a document relative to a collection.

**Transformer:** A neural-network architecture based on attention mechanisms and widely used for contextual language modelling.
