# requirements.md - MediFlow Translation Engine POC

## 1. Project Context and Vision
* **The Problem:** Many immigrants struggle to realize their medical rights, understand medical documents, and communicate with the healthcare services in Israel.
* **The Solution:** Developing a smart, multi-lingual bot (Multi-Agent) that accompanies immigrants and provides full support and accessibility in their native language during interactions with the healthcare system:.
* **Core Function:** The tool must translate medical documents such as ER summaries, doctor summaries, and prescriptions.

## 2. POC Objective
The goal of this phase is to build the complete pipeline for the **translation engine** (MVP):. This covers the entire flow, from evaluating and selecting the translation engine to orchestrating the agents, ensuring transparency, and integrating text extraction (OCR).

## 3. Core Architecture and Components

### A. Evaluation Mechanism (Milestone 1) :
* **Goal:** Develop an evaluation metric pipeline to measure translation quality against a "Gold Standard" official translation.
* **Comet:** Implement a local or cloud pipeline running a HuggingFace Comet model to evaluate the semantic similarity between the machine translation and the reference translation.
* **LLM-as-a-Judge:** Define a strict rubric prompt for Gemini 3 Pro to evaluate translation accuracy on a scale of 1 to 10, with a strong emphasis on medical terms, dosages, and diagnoses, while providing reasoning.
* **Hybrid Testing:** Run the baseline dataset through both evaluators, compare the correlation, and determine the final metric to be fed into DSPy.

### B. Agent Orchestration, Classification, and Training (Milestone 3) :
* **Router Agent:** Develop a lightweight agent (using a model like Gemini 3 Flash) that receives a document and returns only its specific category: Summary, Prescript, Referral, Form, Record.
* **DSPy Signatures:** Define 5 separate signatures with dedicated instructions for each document type (for example, strictly prohibiting the translation of drug names in prescriptions, leaving them in English.
* **Agent Optimization:** Run the DSPy optimizer on each of the 5 agents using the classified dataset and the evaluation mechanism (from Milestone 1) to refine the dedicated prompt for each document template.

### C. Transparency and Security (XAI) (Milestone 4) :
* **Confidence Data Generation:** Modify request settings to extract confidence metrics (Logprobs) from the model for each word/token, or develop a "reflection" function where the model reviews its own translation and returns a list of words it is uncertain about.
* **Tagging Mechanism:** Add code that maps the "weak" words (or marks them in an agreed-upon format like a special Markdown tag) so the UI knows to highlight them (e.g., in yellow or with a warning note).

### D. OCR and UI Integration (Milestone 5) :
* **OCR Integration:** Test and connect an OCR service (such as Google Cloud Vision API - TEXT_DETECTION) to extract Hebrew text from document images and scans.
* **Pipeline Integration:** Connect the OCR to the main backend API pipeline: Image In -> Text Extracted -> Router -> Text Sent to Relevant Agent -> Translated Text Returned: .
* **Frontend UI:** Build a basic web/app interface where the user uploads a file/image, sees a loading icon, and finally receives the translated and formatted document, complete with markings for the "unsafe" words.