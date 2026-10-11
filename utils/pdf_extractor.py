import re
from pypdf import PdfReader


_PDF_ARTIFACT_REPLACEMENTS = {
    "REGUL A TION": "REGULATION",
    "EUR OPEAN": "EUROPEAN",
    "CO UNCIL": "COUNCIL",
    "ar tif icial": "artificial",
    "Provi ders": "Providers",
    "provi ders": "providers",
    "Deplo y ers": "Deployers",
    "deplo y ers": "deployers",
    "high-r isk": "high-risk",
    "High-r isk": "High-risk",
    "syste ms": "systems",
    "syste m": "system",
    "te chnical": "technical",
    "org anisational": "organisational",
    "appropr iate": "appropriate",
    "oblig ations": "obligations",
    "obliga tions": "obligations",
    "author ities": "authorities",
    "author ity": "authority",
    "refe r red": "referred",
    "regist er": "register",
    "marke d": "marked",
    "f or mat": "format",
    "inf or med": "informed",
    "concer ned": "concerned",
    "interac t": "interact",
    "suc h": "such",
    "wa y": "way",
    "the y": "they",
    "ac hieve": "achieve",
    "cybersecur ity": "cybersecurity",
    "sur veillance": "surveillance",
}


def _repair_pdf_spacing(text: str) -> str:
    for bad, good in _PDF_ARTIFACT_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = _repair_pdf_spacing(line.strip())
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract readable PDF text with page markers and light PDF artefact cleanup."""
    reader = PdfReader(file_path)
    pages_text = []

    for index, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        if page_text:
            pages_text.append(f"[PAGE {index}]\n{page_text}")

    return "\n\n".join(pages_text)
