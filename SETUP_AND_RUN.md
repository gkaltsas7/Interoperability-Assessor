# Setup and Run Guide

## Prerequisites

- Windows with Python 3 and pip available through `py`.
- Internet connection and active OpenAI API access.
- A PDF with a readable text layer (no OCR in this prototype).

## Installation (PowerShell)

Open PowerShell in this repository's root folder.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` to set a **real, private** API key:

```dotenv
OPENAI_API_KEY=YOUR_OWN_API_KEY
OPENAI_MODEL=gpt-4.1
```

The public repository contains only `.env.example`, **never** a real `.env` file. If PowerShell script activation is blocked, use `.\.venv\Scripts\python.exe` instead of `python` for the following steps.

Run the application:

```powershell
python -m uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>.

## First assessment

1. Upload a **searchable** PDF.
2. Keep Interoperable Europe Act + Assessment Guidelines selected, with EIF 3.0 enabled.
3. Select the internal assessor lexicon.
4. Choose **Risk Analysis Report** for a short first run, or **Full Assessment Report with Recommendations** for the extended report.
5. Click through Review & Run, generate the report and download the resulting PDF.

The app writes intermediate/diagnostic artifacts into `outputs/<run_id>/`. These folders may include uploaded documents and detailed context; they are ignored by Git and should not be publicly shared without careful review.

## Common issues

- **Missing `OPENAI_API_KEY`**: edit `.env` in the project root; don't put secrets in the code.
- **API 429 / insufficient quota**: check your OpenAI API project's active quota and billing settings.
- **No readable text found in PDF**: the PDF is scanned or otherwise lacks a usable text layer; select a text-based PDF.
- **Server won't start**: verify Python and reinstall dependencies with `python -m pip install -r requirements.txt`.

## Reproducibility notes

The dependencies are pinned in `requirements.txt`. Record the code commit, model name, exact assessed PDF, framework selection, lexicon mode, output language and report type. The system provides reproducible setup and workflow steps, **not guaranteed byte-identical LLM outputs**. Use `sample_outputs/` for an illustrative archived output.
