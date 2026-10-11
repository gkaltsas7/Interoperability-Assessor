# Interoperability Assessor

**AI-assisted interoperability assessment of legal, regulatory, and administrative PDF documents.**

This repository contains the research prototype developed for a diploma thesis at the University of the Aegean. It supports structured assessment against the **Interoperable Europe Act**, its **Guidelines for Interoperability Assessments**, and the **European Interoperability Framework (EIF)**. It is **not** an official EU assessment service and does **not** provide a definitive legal-compliance determination.

## Features

- Browser-based guided assessment workflow (FastAPI backend; HTML/CSS/JavaScript frontend).
- Text extraction from searchable PDF files.
- Staged LLM processing for document profiling, requirement extraction, IEA applicability analysis, service mapping, EIF interoperability layers, risks, and recommendations.
- Stage-specific **keyword/rule-based evidence retrieval** (not embedding-based or vector-database RAG).
- Built-in assessor lexicon with optional custom lexicon upload.
- Internal consistency and citation checks, which **do not replace human validation**.
- PDF export: *Risk Analysis* or *Full Assessment Report with Recommendations*.

## Requirements

- Python 3 and pip (the Python dependency versions are pinned in `requirements.txt`).
- Internet access and an OpenAI API key with an available API quota; API usage may incur charges.
- A PDF with extractable text. The prototype does not provide a full OCR workflow.

## Quick start on Windows (PowerShell)

Open PowerShell in this repository's root folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` locally, set `OPENAI_API_KEY` to **your own** key and leave `OPENAI_MODEL=gpt-4.1` to use the model documented in the thesis. **Never commit `.env` or your key.** Then run:

```powershell
python -m uvicorn app:app --reload
```

Navigate to <http://127.0.0.1:8000>.

> If PowerShell activation is blocked by local execution policy, you can skip activation and use `.\.venv\Scripts\python.exe` in place of `python` for the install and run commands.

More detail: [`SETUP_AND_RUN.md`](SETUP_AND_RUN.md).

## Example run

1. Upload a publicly distributable, text-searchable institutional PDF.
2. Keep the IEA and Guidelines enabled; enable EIF 3.0; leave EIRA disabled unless needed.
3. Select the internal lexicon, English output, and **Full Assessment Report with Recommendations**.
4. Review the settings and run the assessment.
5. Download the PDF report and review requirements, evidence, risks, and recommendations.

See [`sample_outputs/`](sample_outputs/) for one **illustrative, machine-generated report** from the thesis pilot (input document: [University of the Aegean equal-access plan](https://www.aegean.gr/sites/default/files/static/23/05/shedio_pa_gia_atoma_me_e.e.a.a_final.pdf)), and [`docs/Agent_Assessment_Methodology.md`](docs/Agent_Assessment_Methodology.md) for the workflow. The illustrative report has not been independently certified and may include errors; it is provided to document the prototype's actual output.

## Reproducibility

The following are needed to reproduce the **procedure** (not necessarily identical LLM text output):

- Identical input PDF and its text layer.
- Code release/commit and dependencies (`requirements.txt`).
- `OPENAI_MODEL` (default `gpt-4.1`) and availability of the external API.
- Assessment basis and selected versions (IEA, Guidelines, EIF; optional EIRA).
- Report type, output language, lexicon mode, and optional user-supplied guidance.
- When running locally, retain `outputs/<run_id>/assessment_context.json` and the generated PDF for audit/review; **do not upload real output folders** to this repository.

The `domain_knowledge/` directory contains the named EU reference materials as supplied with the research prototype; check the originating institutions' reuse conditions before public redistribution. EU reference material is credited to its respective institutions and is not claimed as original work.

## Project layout

```text
app.py                  FastAPI app and report-generation endpoint
frontend/               Browser UI
utils/                  PDF extraction, evidence retrieval, pipeline, LLM and reports
prompt/ and prompts/    Master instructions and documented staged prompts
lexicon/                Controlled assessor lexicon
requirements_docs/      Implementation requirements
domain_knowledge/       Assessment reference PDFs and optional EIRA resources
sample_outputs/         Example generated PDF report
SETUP_AND_RUN.md        Local setup instructions
DEMO_GUIDE.md            Guided demo notes
requirements.txt        Python dependencies and versions
.env.example            Safe placeholder configuration
```

## Safety and limitations

- Generated analyses are **decision-support**, not legal advice or compliance certification.
- Users must independently verify source citations, binding status, legal applicability and assumptions underlying recommendations.
- Do not upload personal, confidential or sensitive documents to an external API without appropriate institutional authorization and data-protection review.
- The current prototype is not a production-grade multi-user service.

## Academic context

Diploma-thesis implementation at the University of the Aegean, 2026. See the thesis volume for methodology, corpus composition, full evaluation, limitations and references.

No open-source license is included automatically. Publication and reuse permissions should be confirmed by the repository owner and relevant institutional stakeholders.
