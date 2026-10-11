# Demo Guide — How to Explain the Tool

## One-sentence explanation

The tool is a configurable AI-enabled Interoperability Assessment Assessor: it takes a legal or policy document, applies Interoperable Europe assessment logic, checks sensitive drafting patterns and generates either a risk-focused report or a full report with recommendations.

## What to show first

1. Open the UI at `http://127.0.0.1:8000`.
2. Explain that the interface is organised as a guided workflow, not as one long form.
3. Go through the tabs:
   - Document
   - Assessment Basis
   - Sensitive Terms
   - Report Type
   - Review & Run
4. Upload a PDF document.
5. Show the assessment-basis choices.
6. Show the sensitive-terms options.
7. Show the two report types.
8. Review the configuration summary.
9. Run a short assessment or show a sample output if needed.

## Key concepts

### UI logic

The v6 UI is designed as a guided assessment interface. The aim is to make the tool understandable to a user who may not already know the details of interoperability assessment. Each tab explains what the user is selecting and why it matters.

### Assessment basis

The user can select the documents/frameworks used as the basis of the assessment. The Interoperable Europe Act and IE Act Guidelines are locked. EIF 3.0 is enabled by default. EIRA is optional. EIF 4.0 draft is pending until the source file is provided.

### Sensitive terms

The lexicon is a controlled vocabulary, not a list of forbidden words. A detected term is a signal for contextual assessment. The internal lexicon has two parts: interoperability-sensitive terms and digital-transformation-sensitive terms.

### Report types

- Risk Analysis Report: shorter output focused on trigger logic, binding requirements, affected services, sensitive terms and key risks.
- Full Assessment Report with Recommendations: broader report including corrective measures, drafting recommendations, implementation reasoning, validation and traceability.

### Domain knowledge

The official reference files are stored in the project for traceability and reproducibility. The IEA / Guidelines / EIF logic is encoded in the assessment prompts and pipeline. The sensitive lexicon is read actively at runtime.

### Prompt structure

The active master prompt is `prompt/prompt_v2.txt`. The current implementation keeps most stage-specific instructions inside `utils/assessment_pipeline.py`. The `prompts/` folder documents the modular prompt architecture and can be used for future refactoring.

### Excel lexicon vs custom uploads

The official internal lexicon is the Excel file `lexicon/sensitive_terms_lexicon.xlsx`. The tool can also read custom user-uploaded lists in `.xlsx`, `.csv`, `.txt` or `.json`. At runtime, all formats are normalised into the internal structure needed by the assessment pipeline.

## Safe answer for API key / connection check

The OpenAI API key is stored in `.env` as `OPENAI_API_KEY` and loaded by `utils/llm_engine.py`. The assessment pipeline calls `llm_engine.py` whenever it needs an LLM response.

## Safe answer for API quota issue

The application reaches the OpenAI API call, but a `429 - insufficient_quota` response means the configured API project/key should be checked for quota, billing or usage limits. It is not, by itself, evidence of a local code error.

## Safe answer for production readiness

The tool is suitable as a research/thesis application prototype demonstrating the assessment workflow, configurable inputs, lexicon use and report generation. A production deployment would require additional deployment, authentication, logging, monitoring and cost-optimisation work.

## Files to know

- `app.py` — FastAPI backend entry point
- `frontend/index.html` — guided UI
- `utils/llm_engine.py` — OpenAI connection layer
- `utils/assessment_pipeline.py` — core assessment logic
- `utils/lexicon_engine.py` — sensitive terms loading and detection
- `utils/pdf_extractor.py` — PDF text extraction
- `utils/evidence_retriever.py` — stage-specific evidence retrieval
- `utils/report_generator.py` — Word report generation
