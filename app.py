from pathlib import Path
import json
import shutil
from datetime import datetime
from uuid import uuid4
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from utils.pdf_extractor import extract_text_from_pdf
from utils.llm_engine import load_prompt
from utils.assessment_pipeline import run_assessment_pipeline
from utils.report_generator import save_report_to_docx, save_report_to_pdf
from utils.lexicon_engine import assess_uploaded_guideline_relevance, load_structured_lexicon

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompt" / "prompt_v2.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
DOMAIN_KNOWLEDGE_DIR = BASE_DIR / "domain_knowledge"

app = FastAPI(title="Configurable Interoperability Assessment Platform", debug=True)
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend"), name="static")


@app.get("/")
async def home():
    return FileResponse(BASE_DIR / "frontend" / "index.html")


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _save_optional_upload(upload: Optional[UploadFile], destination: Path) -> Optional[Path]:
    if upload is None or not upload.filename:
        return None
    destination.mkdir(parents=True, exist_ok=True)
    safe_name = Path(upload.filename).name.replace("/", "_").replace("\\", "_")
    out_path = destination / safe_name
    with out_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return out_path


def _sample_text_from_uploaded_basis(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return extract_text_from_pdf(str(path))[:6000]
        if suffix in {".txt", ".md", ".csv", ".json"}:
            return path.read_text(encoding="utf-8", errors="ignore")[:6000]
    except Exception:
        return ""
    return ""


@app.post("/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    report_type: str = Form("risk_analysis"),
    assessment_mode: str = Form(""),  # legacy compatibility with previous UI
    output_language: str = Form("en"),
    basis_eif3: str = Form("true"),
    basis_eira: str = Form("false"),
    basis_eif4: str = Form("false"),
    lexicon_mode: str = Form("internal"),
    use_internal_lexicon: str = Form("true"),  # legacy compatibility
    use_interoperability_lexicon: str = Form("true"),  # legacy compatibility
    use_digital_transformation_lexicon: str = Form("true"),  # legacy compatibility
    custom_guideline: Optional[UploadFile] = File(None),
    custom_lexicon: Optional[UploadFile] = File(None),
):
    if file.content_type not in {"application/pdf", "application/octet-stream"} and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted as the main assessment document.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = OUTPUT_DIR / run_id
    uploads_dir = run_dir / "uploads"
    run_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    temp_pdf_path = run_dir / "uploaded_document.pdf"
    report_docx_path = run_dir / "interoperability_assessment_report.docx"
    report_pdf_path = run_dir / "interoperability_assessment_report.pdf"
    context_json_path = run_dir / "assessment_context.json"
    report_txt_path = run_dir / "interoperability_assessment_report.txt"

    with temp_pdf_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    custom_guideline_path = _save_optional_upload(custom_guideline, uploads_dir / "custom_guidelines")
    custom_lexicon_path = _save_optional_upload(custom_lexicon, uploads_dir / "custom_lexicons")

    document_text = extract_text_from_pdf(str(temp_pdf_path))
    if not document_text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in the PDF.")

    prompt_text = load_prompt(str(PROMPT_PATH))

    selected_report_type = (report_type or assessment_mode or "risk_analysis").strip().lower()
    # Backward compatibility with older Standard/Enhanced values.
    if selected_report_type in {"standard_assessment", "standard", "institutional_assessment"}:
        selected_report_type = "risk_analysis"
    if selected_report_type in {"enhanced_assessment", "enhanced", "expert_assessment", "full_assessment"}:
        selected_report_type = "full_assessment"

    custom_guideline_sample = _sample_text_from_uploaded_basis(custom_guideline_path)
    guideline_relevance = assess_uploaded_guideline_relevance(custom_guideline_path, custom_guideline_sample)
    guideline_status = str(guideline_relevance.get("status", "not_uploaded")) if isinstance(guideline_relevance, dict) else "not_uploaded"
    guideline_is_usable = guideline_status in {"relevant", "partially_relevant"}

    selected_lexicon_mode = (lexicon_mode or "internal").strip().lower()
    if selected_lexicon_mode not in {"internal", "custom"}:
        # Backward compatibility with old checkboxes.
        selected_lexicon_mode = "internal" if _truthy(use_internal_lexicon) else "custom"

    custom_lexicon_summary = {"uploaded": False, "entries_loaded": 0, "filename": None}
    custom_lexicon_paths = []
    if custom_lexicon_path:
        entries = load_structured_lexicon(custom_lexicon_path, source="user_uploaded")
        custom_lexicon_summary = {
            "uploaded": True,
            "entries_loaded": len(entries),
            "filename": custom_lexicon_path.name,
            "format_note": "Simple lists, partial tables and full lexicon format are accepted and normalised.",
        }
        custom_lexicon_paths.append(str(custom_lexicon_path))

    if selected_lexicon_mode == "custom" and not custom_lexicon_paths:
        raise HTTPException(status_code=400, detail="Custom lexicon mode selected, but no lexicon file was uploaded.")

    assessment_basis = {
        "interoperable_europe_act": {"selected": True, "locked": True, "path": str(DOMAIN_KNOWLEDGE_DIR / "legal_basis" / "01 Interoperability Act_EN_TXT.pdf")},
        "iea_guidelines": {"selected": True, "locked": True, "path": str(DOMAIN_KNOWLEDGE_DIR / "legal_basis" / "03 Guidelines for Interoperability Assessments December 2024_v5.pdf")},
        "eif_3_0": {"selected": _truthy(basis_eif3), "locked": False, "path": str(DOMAIN_KNOWLEDGE_DIR / "legal_basis" / "02 EIF 30_brochure_final.pdf")},
        "eira_6_1_0": {"selected": _truthy(basis_eira), "locked": False, "path": str(DOMAIN_KNOWLEDGE_DIR / "frameworks" / "eira" / "EIRA 6.1.0 Release.zip")},
        "eif_4_0_draft": {"selected": _truthy(basis_eif4), "locked": False, "path": "not_supplied_yet"},
        "custom_uploaded_guideline": {
            "selected": guideline_is_usable,
            "uploaded": custom_guideline_path is not None,
            "path": str(custom_guideline_path) if custom_guideline_path else None,
            "relevance": guideline_relevance,
        },
    }

    assessment_config = {
        "document_category": "auto_detect",
        "assessment_mode": selected_report_type,
        "report_type": selected_report_type,
        "assessment_focus": "auto_detect",
        "evidence_approach": "strict",
        "output_language": output_language,
        "assessment_basis": assessment_basis,
        "lexicon_config": {
            "lexicon_mode": selected_lexicon_mode,
            "use_internal_lexicon": selected_lexicon_mode == "internal",
            # The internal lexicon still contains interoperability and digital-transformation categories,
            # but the UI exposes them as one assessor lexicon rather than two separate user choices.
            "use_interoperability_lexicon": True,
            "use_digital_transformation_lexicon": True,
            "custom_lexicon_paths": custom_lexicon_paths,
            "custom_lexicon_summary": custom_lexicon_summary,
        },
    }

    try:
        result = run_assessment_pipeline(
            prompt_text=prompt_text,
            document_text=document_text,
            assessment_config=assessment_config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Assessment generation failed: {exc}") from exc

    report_content = result["final_report"]
    context = result["context"]
    context["run_id"] = run_id
    context["uploaded_files"] = {
        "main_document": file.filename,
        "custom_guideline": custom_guideline_path.name if custom_guideline_path else None,
        "custom_lexicon": custom_lexicon_path.name if custom_lexicon_path else None,
    }

    report_txt_path.write_text(report_content, encoding="utf-8")
    context_json_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep DOCX internally for audit/editing, but do not let an internal DOCX
    # formatting problem block the user-facing PDF export.
    try:
        save_report_to_docx(report_text=report_content, output_path=str(report_docx_path))
    except Exception as exc:
        (run_dir / "docx_generation_error.txt").write_text(str(exc), encoding="utf-8")

    # Export PDF directly as the user-facing output. This pure-Python path does
    # not require LibreOffice/soffice on the machine.
    try:
        save_report_to_pdf(report_text=report_content, output_path=str(report_pdf_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Assessment report was generated but PDF rendering failed: {exc}") from exc

    download_name = "risk_analysis_report.pdf" if selected_report_type == "risk_analysis" else "full_assessment_report_with_recommendations.pdf"
    return FileResponse(
        path=report_pdf_path,
        filename=download_name,
        media_type="application/pdf",
    )
