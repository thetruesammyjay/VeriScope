# Requirements-Gathering Questionnaire

## Automated Fake News Detection Using Natural Language Processing

### Participant Information

This questionnaire supports the requirements-gathering phase of an academic study on the design of an automated fake-news detection system using Natural Language Processing (NLP). The proposed system will analyse submitted news text, produce a cautious classification, and search current sources for evidence that may support or contradict important claims.

The purpose of this questionnaire is to understand how people encounter, assess, and share online news, and what they would expect from such a system. Responses will be used to identify functional and non-functional requirements. Participation is voluntary. Do not provide passwords, bank details, government identification numbers, or other sensitive personal information. The questionnaire should take approximately 7–10 minutes.

Unless a question says otherwise, select one answer. For questions marked **Select all that apply**, more than one answer may be selected.

---

## Section A: Consent and Participant Profile

### 1. Consent

**Question type:** Multiple choice — required

Do you voluntarily agree to participate in this academic requirements-gathering questionnaire?

- Yes, I agree to participate.
- No, I do not agree to participate. *(End the questionnaire.)*

### 2. Relationship with online news

**Question type:** Multiple choice — required

Which option best describes you?

- Member of the general public or news reader
- Student or researcher
- Journalist, editor, or media worker
- Teacher or lecturer
- Public-sector, civil-society, or community worker
- Technology or information-services professional
- Other: __________

### 3. Age group

**Question type:** Multiple choice — optional

- Under 18
- 18–24
- 25–34
- 35–44
- 45–54
- 55 or above
- Prefer not to say

### 4. Country or region of residence

**Question type:** Short answer — optional

Please state your country or broad region of residence. Do not provide a street address.

### 5. Frequency of online-news use

**Question type:** Multiple choice — required

How often do you read or receive news through digital channels?

- Several times a day
- Once a day
- A few times a week
- About once a week
- Less than once a week
- Almost never

---

## Section B: News Access and Verification Practices

### 6. News channels used

**Question type:** Checkboxes — select all that apply

Where do you usually encounter news?

- Online newspapers or news websites
- Social-media platforms
- Messaging applications or group chats
- Radio or television websites and applications
- Blogs or independent websites
- Video-sharing platforms
- Printed newspapers or magazines
- Radio or television broadcasts
- Friends, relatives, or colleagues
- Other: __________

### 7. Topics commonly encountered

**Question type:** Checkboxes — select all that apply

Which topics do you commonly read or receive online?

- Politics and elections
- Public health
- Security and conflict
- Business and finance
- Education
- Science and technology
- Sports
- Entertainment
- Religion or community affairs
- International news
- Other: __________

### 8. What makes a report appear trustworthy?

**Question type:** Checkboxes — select all that apply

Which features make you more likely to trust a news report?

- A named author or news organisation
- A publication date and clear context
- Links to original documents or evidence
- Agreement with several independent sources
- Quotes from identifiable people or institutions
- Professional writing and presentation
- A familiar website or journalist
- Images, video, or other supporting material
- Recommendations from people I trust
- A correction or editorial policy is visible
- Other: __________

### 9. How do you verify a doubtful report?

**Question type:** Checkboxes — select all that apply

When a report seems doubtful, what do you usually do?

- Search for the claim on a search engine
- Check an official government or organisational website
- Compare reports from multiple news organisations
- Consult a fact-checking organisation
- Check the author, website, or publication history
- Ask another person
- Check the date, images, or video separately
- Do nothing and avoid sharing it
- Share it with a warning or question
- I usually do not verify reports
- Other: __________

### 10. Difficulties when verifying information

**Question type:** Checkboxes — select all that apply

What makes verification difficult for you?

- Too many search results
- Difficulty identifying reliable sources
- Conflicting reports
- Sources are unavailable or blocked
- Information is outdated
- The original source is unclear
- The claim contains several different assertions
- Lack of time
- Technical or internet-access limitations
- Difficulty understanding specialist language
- Uncertainty about whether a source is independent
- I have not experienced significant difficulty
- Other: __________

### 11. Experience with misinformation

**Question type:** Multiple choice — required

Have you ever received, believed, or shared information that was later shown to be false or misleading?

- Yes, I received it but did not share it
- Yes, I shared it before discovering the problem
- Yes, I believed it for some time
- I am not sure
- No
- Prefer not to say

### 12. Consequences of false or misleading reports

**Question type:** Checkboxes — select all that apply

What effects can false or misleading news have?

- Confusion or unnecessary fear
- Damage to a person’s or organisation’s reputation
- Poor health or safety decisions
- Financial loss
- Political tension or conflict
- Discrimination or harassment
- Reduced trust in legitimate journalism or institutions
- Unnecessary sharing and public discussion
- I do not think it has significant effects
- Other: __________

---

## Section C: Requirements for an Automated Detection System

For Questions 13–17, use the following scale where applicable:

**1 = Not important, 2 = Slightly important, 3 = Moderately important, 4 = Very important, 5 = Extremely important.**

### 13. Useful input methods

**Question type:** Checkboxes — select all that apply

How should a proposed system allow users to submit information?

- Paste a headline
- Paste a news article
- Paste a social-media post
- Paste a message received in a chat application
- Enter a web-page URL
- Upload a text document
- Select a language
- Submit a claim separately from the full article
- Other: __________

### 14. Desired prediction output

**Question type:** Checkboxes — select all that apply

Which information should the system display after analysing text?

- Likely real or likely fake classification
- Confidence score
- Short explanation of the prediction
- Important claims identified in the text
- Evidence that supports the claims
- Evidence that contradicts the claims
- An insufficient-evidence result
- Links to the sources used
- Publication dates of the sources
- Model name and version
- Processing time
- Warning that the result is not a final fact-check
- Other: __________

### 15. Importance of proposed system functions

**Question type:** Linear-scale grid — required

How important are the following functions?

| Proposed function | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Submit a headline or article for analysis |  |  |  |  |  |
| Reject empty, extremely short, or excessively large input |  |  |  |  |  |
| Extract checkable claims from an article |  |  |  |  |  |
| Analyse several claims separately |  |  |  |  |  |
| Search current online sources |  |  |  |  |  |
| Rank evidence by relevance and source quality |  |  |  |  |  |
| Display supporting and contradicting evidence separately |  |  |  |  |  |
| Show an insufficient-evidence result when appropriate |  |  |  |  |  |
| Preserve links and dates for retrieved sources |  |  |  |  |  |
| Allow the user to copy or export the analysis |  |  |  |  |  |

### 16. Preferred prediction wording

**Question type:** Multiple choice — required

Which wording would be clearest for the text-classification result?

- Likely real / likely fake
- Real / fake
- Reliable / unreliable
- Supported / contradicted
- A numerical score only
- Other: __________

### 17. Acceptable response time

**Question type:** Multiple choice — required

How long would you consider acceptable for a result that includes online evidence?

- Less than 5 seconds
- 5–15 seconds
- 16–30 seconds
- 31–60 seconds
- More than one minute, if the evidence is useful
- I would prefer a quick preliminary result followed by evidence

### 18. Handling uncertain results

**Question type:** Multiple choice — required

What should the system do when it cannot find enough reliable evidence?

- Clearly state that there is insufficient evidence
- Provide the prediction but show a strong warning
- Ask the user to submit a shorter or clearer claim
- Return the available sources without a prediction
- Attempt another search automatically
- Other: __________

---

## Section D: Current-Source Retrieval and Evidence Verification

### 19. Preferred evidence sources

**Question type:** Checkboxes — select all that apply

Which sources should receive priority when the system searches for evidence?

- Official government websites
- Official statements from organisations
- Established news organisations
- Academic or research institutions
- Professional associations
- Fact-checking organisations
- Local or regional news organisations
- International news organisations
- Social-media posts from verified accounts
- Any source that directly discusses the claim
- Other: __________

### 20. Source information to display

**Question type:** Checkboxes — select all that apply

What information would help you judge a retrieved source?

- Source name
- Article title
- URL
- Publication date
- Date retrieved
- Author or organisation
- Short relevant passage
- Explanation of how the passage relates to the claim
- Whether other independent sources agree
- Possible conflict of interest or source limitation
- Other: __________

### 21. Conflicting evidence

**Question type:** Multiple choice — required

If reliable sources disagree, how should the system report the result?

- Display both positions and label the result as mixed evidence
- Select the source with the highest reliability score
- Return insufficient evidence
- Ask the user to decide which source to trust
- Do not display conflicting sources
- Other: __________

### 22. Source independence

**Question type:** Multiple choice — required

How important is it for the system to identify when several websites repeat the same original report?

- Not important
- Slightly important
- Moderately important
- Very important
- Extremely important

### 23. Freshness of evidence

**Question type:** Multiple choice — required

When a claim concerns a current event, how important is the publication date of evidence?

- Not important
- Slightly important
- Moderately important
- Very important
- Extremely important

---

## Section E: Trust, Privacy, Accessibility, and Responsible Use

### 24. Factors that would increase trust

**Question type:** Checkboxes — select all that apply

What would make you trust the system’s results more?

- Clear explanation of how the result was produced
- Links to the original evidence
- Visible source dates
- More than one independent source
- A statement of uncertainty
- Ability to report an incorrect result
- Published evaluation results
- Independent review or institutional approval
- Visible model version and update date
- Protection of submitted text and personal information
- Other: __________

### 25. Privacy expectations

**Question type:** Checkboxes — select all that apply

Which privacy safeguards would you expect?

- Do not collect a name unless necessary
- Do not collect an email address unless necessary
- Do not permanently store submitted text by default
- Explain how submitted text is used
- Allow users to delete submitted text
- Do not use submissions for unrelated advertising
- Encrypt data during transmission
- Restrict access to stored analysis records
- Provide a privacy policy
- Other: __________

### 26. Accessibility and connectivity

**Question type:** Checkboxes — select all that apply

Which considerations would make the system easier to use?

- Mobile-friendly interface
- Low-data mode
- Clear and simple language
- Readable colour contrast
- Keyboard accessibility
- Support for screen readers
- Results that remain understandable on a small screen
- Ability to analyse text when internet connectivity is unstable
- Support for local or widely used languages in future versions
- Other: __________

### 27. Responsible-use boundaries

**Question type:** Checkboxes — select all that apply

Which uses of an automated prediction should be prohibited or treated with caution?

- Automatically deleting or censoring content
- Automatically suspending a user account
- Determining legal liability
- Making employment or school disciplinary decisions
- Replacing professional journalists or fact-checkers
- Treating a high confidence score as proof of truth
- Publishing a result without showing its limitations
- Using the system to target individuals or groups
- None of these concerns
- Other: __________

### 28. Likelihood of use

**Question type:** Multiple choice — required

If available, how likely would you be to use a system that provides a prediction together with current-source evidence?

- Very likely
- Likely
- Not sure
- Unlikely
- Very unlikely

---

## Section F: Open-Ended Requirements

### 29. Most important problem

**Question type:** Paragraph — optional

What is the most important problem an automated fake-news detection system should solve for you?

### 30. Essential feature

**Question type:** Paragraph — optional

What feature would make the proposed system genuinely useful to you?

### 31. Concern or limitation

**Question type:** Paragraph — optional

What is your biggest concern about relying on an automated prediction and online evidence?

### 32. Additional requirements

**Question type:** Paragraph — optional

Please describe any other requirement, improvement, or recommendation the researcher should consider.

---

## Researcher Administration Notes

- Do not collect names, passwords, payment details, identification numbers, or precise addresses.
- Keep consent as the first required question. A “No” response should submit the form without showing the remaining sections.
- Questions 2–5 support participant profiling; Questions 6–12 identify the existing problem; Questions 13–23 identify functional requirements; Questions 24–28 identify non-functional and responsible-use requirements; and Questions 29–32 capture additional requirements.
- Report responses in aggregate. Do not identify individual participants in the dissertation.
- Before deployment, obtain the required institutional or supervisor approval and provide the approved participant-information and consent wording.
- A questionnaire response expresses a user requirement or perception; it is not evidence that a news report is true or false.
