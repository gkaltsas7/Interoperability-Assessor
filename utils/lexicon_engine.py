import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
STRUCTURED_XLSX_PATH = BASE_DIR / "lexicon" / "sensitive_terms_lexicon.xlsx"

FULL_COLUMNS = [
    "id", "term", "language", "category", "subcategory", "severity", "EIF_dimension",
    "risk_type", "explanation", "recommended_control", "example_context", "matching_type", "active", "notes"
]

DEFAULT_ENTRY = {
    "language": "unknown",
    "category": "user_uploaded_uncategorised",
    "subcategory": "unspecified",
    "severity": "medium",
    "EIF_dimension": "mixed",
    "risk_type": "contextual_review",
    "explanation": "User-provided sensitive term. It should be assessed in context and does not automatically indicate non-compliance.",
    "recommended_control": "Review the provision and assess whether it creates interoperability or digital-transformation risk.",
    "matching_type": "phrase",
    "active": "yes",
    "notes": "Auto-normalised from a simple or partial user-uploaded list.",
}


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _normalise_entry(raw: Dict[str, Any], source: str = "internal") -> Optional[Dict[str, str]]:
    mapped: Dict[str, Any] = {}
    for k, v in raw.items():
        key = _normalise_key(k)
        if key in {"word", "phrase", "term_phrase", "term_or_phrase", "sensitive_word", "sensitive_phrase", "pattern"}:
            key = "term"
        if key in {"eif", "dimension", "interoperability_dimension", "eif_dimension"}:
            key = "EIF_dimension"
        if key in {"why_sensitive", "why_it_matters", "why_problematic", "reason", "description"}:
            key = "explanation"
        if key in {"suggested_improvement", "suggested_control", "control", "correction", "recommendation", "recommended_correction"}:
            key = "recommended_control"
        mapped[key] = v

    term = str(mapped.get("term") or "").strip()
    if not term or term.lower() in {"term", "word", "phrase", "none", "nan"}:
        return None

    entry = {**DEFAULT_ENTRY}
    for col in FULL_COLUMNS:
        norm_col = _normalise_key(col)
        if col in mapped and str(mapped[col]).strip():
            entry[col] = str(mapped[col]).strip()
        elif norm_col in mapped and str(mapped[norm_col]).strip():
            entry[col] = str(mapped[norm_col]).strip()
    entry["term"] = term
    entry["source"] = source
    if not entry.get("id"):
        entry["id"] = f"USER-{abs(hash((source, term))) % 100000:05d}"
    if not entry.get("category") or entry.get("category") == "user_uploaded_uncategorised":
        entry.update(_auto_classify(term, entry))
    return {k: str(v) for k, v in entry.items() if v is not None}


def _auto_classify(term: str, entry: Dict[str, Any]) -> Dict[str, str]:
    t = term.lower()
    digital = [
        "paper", "printed", "hard copy", "physical", "in person", "presence", "stamp", "seal", "fax", "post",
        "newspaper", "press", "public book", "manual", "original document", "certified copy",
        "χαρτ", "έντυπ", "φυσική", "αυτοπρόσω", "σφραγ", "φαξ", "ταχυδρο", "εφημερί", "τύπο",
        "δημόσιο βιβλίο", "πρωτότυπο", "επικυρωμένο", "πρόσφορο μέσο", "καθυστέρηση"
    ]
    interop = [
        "closed", "proprietary", "national-only", "national only", "domestic registry", "specific software",
        "single database", "single central", "data model", "interconnection", "authority-defined", "technical specifications",
        "κλειστό", "ενιαία βάση", "εθνικό μητρώο", "τεχνικές προδιαγραφές"
    ]
    if any(x in t for x in digital):
        return {
            "category": "digital_transformation_sensitive",
            "subcategory": entry.get("subcategory") or "auto_classified",
            "risk_type": entry.get("risk_type") or "digital_transformation_risk",
            "EIF_dimension": entry.get("EIF_dimension") or "user_centricity / technical",
        }
    if any(x in t for x in interop):
        return {
            "category": "interoperability_sensitive",
            "subcategory": entry.get("subcategory") or "auto_classified",
            "risk_type": entry.get("risk_type") or "interoperability_risk",
            "EIF_dimension": entry.get("EIF_dimension") or "semantic / technical",
        }
    return {}


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings = []
    for si in root.findall("a:si", ns):
        texts = [t.text or "" for t in si.findall(".//a:t", ns)]
        strings.append("".join(texts))
    return strings


def _xlsx_sheet_paths(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    # Minimal XLSX reader: returns sheet names and paths. Sufficient for our lexicon/source tables.
    wb_xml = ET.fromstring(zf.read("xl/workbook.xml"))
    rels_xml = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    rid_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_xml.findall("rel:Relationship", ns)}
    sheets = []
    for sh in wb_xml.findall("a:sheets/a:sheet", ns):
        name = sh.attrib.get("name", "Sheet")
        rid = sh.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rid_to_target.get(rid, "")
        if target:
            target = target.lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            sheets.append((name, target))
    return sheets


def _cell_text(cell: ET.Element, shared: List[str], ns: Dict[str, str]) -> str:
    typ = cell.attrib.get("t")
    if typ == "inlineStr":
        texts = [t.text or "" for t in cell.findall(".//a:t", ns)]
        return "".join(texts).strip()
    v = cell.find("a:v", ns)
    if v is None or v.text is None:
        return ""
    if typ == "s":
        try:
            return shared[int(v.text)]
        except Exception:
            return ""
    return v.text.strip()


def _read_xlsx_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            shared = _read_shared_strings(zf)
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for sheet_name, sheet_path in _xlsx_sheet_paths(zf):
                if "metadata" in sheet_name.lower() or "instruction" in sheet_name.lower():
                    continue
                try:
                    root = ET.fromstring(zf.read(sheet_path))
                except KeyError:
                    continue
                table: List[List[str]] = []
                for row in root.findall(".//a:sheetData/a:row", ns):
                    values: Dict[int, str] = {}
                    max_col = 0
                    for c in row.findall("a:c", ns):
                        ref = c.attrib.get("r", "A1")
                        letters = re.match(r"([A-Z]+)", ref)
                        col_idx = 0
                        if letters:
                            for ch in letters.group(1):
                                col_idx = col_idx * 26 + (ord(ch) - ord("A") + 1)
                            col_idx -= 1
                        values[col_idx] = _cell_text(c, shared, ns)
                        max_col = max(max_col, col_idx)
                    if values:
                        table.append([values.get(i, "") for i in range(max_col + 1)])
                if not table:
                    continue
                # Find the actual header row. Readable workbooks may have title/instruction rows
                # before the table header, so do not assume the first non-empty row is the header.
                known_headers = {
                    "term", "term_phrase", "term_or_phrase", "word", "phrase",
                    "language", "severity", "why_sensitive", "why_it_matters",
                    "suggested_improvement", "suggested_control", "notes", "category"
                }
                header_idx = -1
                for idx, candidate in enumerate(table[:20]):
                    normalised_candidate = {_normalise_key(x) for x in candidate if str(x).strip()}
                    if normalised_candidate & known_headers and (
                        {"term", "term_phrase", "term_or_phrase", "word", "phrase"} & normalised_candidate
                    ):
                        header_idx = idx
                        break
                if header_idx < 0:
                    header_idx = 0
                    while header_idx < len(table) and not any(str(x).strip() for x in table[header_idx]):
                        header_idx += 1
                if header_idx >= len(table):
                    continue
                headers = [_normalise_key(x) or f"col_{i}" for i, x in enumerate(table[header_idx])]
                # If no obvious term column, treat first non-empty column as term.
                has_term = "term" in headers or any(h in {"term_phrase", "term_or_phrase", "word", "phrase", "sensitive_word", "sensitive_phrase"} for h in headers)
                for row in table[header_idx + 1:]:
                    if not any(str(x).strip() for x in row):
                        continue
                    if has_term:
                        raw = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
                    else:
                        raw = {"term": next((x for x in row if str(x).strip()), "")}
                    raw["_sheet"] = sheet_name
                    rows.append(raw)
    except Exception:
        return []
    return rows


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    sample = text[:2048]
    dialect = csv.Sniffer().sniff(sample) if "," in sample or ";" in sample or "\t" in sample else csv.excel
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    first = [x.strip() for x in next(csv.reader([lines[0]], dialect=dialect))]
    has_header = any(_normalise_key(x) in {"term", "word", "phrase", "category", "severity"} for x in first)
    if has_header:
        reader = csv.DictReader(lines, dialect=dialect)
        return [dict(r) for r in reader]
    return [{"term": line.strip()} for line in lines]


def _read_json_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x if isinstance(x, dict) else {"term": x} for x in data]
    if isinstance(data, dict):
        rows: List[Dict[str, Any]] = []
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        rows.append({"category": key, **item})
                    else:
                        rows.append({"category": key, "term": item})
        if rows:
            return rows
    return []


def _read_text_rows(path: Path) -> List[Dict[str, Any]]:
    return [{"term": line.strip()} for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def load_structured_lexicon(path: Path = STRUCTURED_XLSX_PATH, source: str = "internal") -> List[Dict[str, str]]:
    if not path or not Path(path).exists():
        return []
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        raw_rows = _read_xlsx_rows(path)
    elif ext == ".json":
        raw_rows = _read_json_rows(path)
    elif ext in {".csv", ".tsv"}:
        raw_rows = _read_csv_rows(path)
    else:
        raw_rows = _read_text_rows(path)
    entries = []
    seen = set()
    for raw in raw_rows:
        e = _normalise_entry(raw, source=source)
        if not e:
            continue
        key = (e.get("term", "").lower(), e.get("category", "").lower(), source)
        if key not in seen:
            seen.add(key)
            entries.append(e)
    return entries


def _matches(text: str, term: str, matching_type: str = "phrase") -> bool:
    if not term:
        return False
    if matching_type == "exact":
        return text.strip().lower() == term.lower()
    if matching_type == "contains":
        return term.lower() in text.lower()
    # Semantic terms still receive direct phrase matching here; LLM receives unmatched config separately via stage prompt if needed.
    pattern = r"(?i)(?<!\w)" + re.escape(term) + r"(?!\w)"
    return bool(re.search(pattern, text)) or term.lower() in text.lower()


def _find_evidence(text: str, term: str, window: int = 140) -> str:
    match = re.search(re.escape(term), text, flags=re.IGNORECASE)
    if not match:
        return "Missing Information"
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    return " ".join(text[start:end].split())


def detect_antipatterns(document_text: str, lexicon_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Detect sensitive terms from internal and optional uploaded lexicons.

    The active internal lexicon is a readable Excel controlled vocabulary. It can
    be simple (Term / Phrase, Severity, Why sensitive, Suggested improvement) or
    richer. During runtime the parser normalises it into the common internal
    structure needed by the assessment pipeline. User lists may be one-column
    simple term lists, partial structured tables or the full internal format.
    """
    lexicon_config = lexicon_config or {}
    use_internal = bool(lexicon_config.get("use_internal_lexicon", True))
    use_interop = bool(lexicon_config.get("use_interoperability_lexicon", True))
    use_digital = bool(lexicon_config.get("use_digital_transformation_lexicon", True))
    custom_paths = lexicon_config.get("custom_lexicon_paths") or []
    if isinstance(custom_paths, (str, Path)):
        custom_paths = [custom_paths]

    entries: List[Dict[str, str]] = []
    if use_internal:
        entries.extend(load_structured_lexicon(STRUCTURED_XLSX_PATH, source="internal_structured"))
    for p in custom_paths:
        try:
            entries.extend(load_structured_lexicon(Path(p), source="user_uploaded"))
        except Exception:
            continue

    filtered: List[Dict[str, str]] = []
    for e in entries:
        active = str(e.get("active", "yes")).strip().lower()
        if active in {"no", "false", "0", "inactive"}:
            continue
        cat = str(e.get("category", "")).lower()
        if "interoperability" in cat and not use_interop:
            continue
        if "digital_transformation" in cat and not use_digital:
            continue
        filtered.append(e)

    normalized = " ".join(str(document_text or "").split())
    findings: List[Dict[str, str]] = []
    seen = set()
    for e in filtered:
        term = e.get("term", "")
        if not _matches(normalized, term, e.get("matching_type", "phrase").lower()):
            continue
        key = (term.lower(), e.get("category", "").lower())
        if key in seen:
            continue
        seen.add(key)
        findings.append({
            "id": e.get("id", ""),
            "category": e.get("category", "user_uploaded"),
            "subcategory": e.get("subcategory", "unspecified"),
            "detected_wording": term,
            "severity": e.get("severity", "medium"),
            "eif_dimension": e.get("EIF_dimension", e.get("eif_dimension", "mixed")),
            "risk_type": e.get("risk_type", "contextual_review"),
            "why_problematic": e.get("explanation", "May create interoperability friction."),
            "evidence_excerpt": _find_evidence(normalized, term),
            "micro_correction": e.get("recommended_control", "Use clearer digital-ready wording."),
            "macro_digital_by_default_replacement": e.get("recommended_control", "Define a digital-ready interoperable service channel."),
            "source": e.get("source", "internal"),
        })

    return {
        "antipattern_findings": findings[:50],
        "lexicon_summary": {
            "total_entries_loaded": len(entries),
            "entries_after_filtering": len(filtered),
            "matches": len(findings),
            "custom_lists_used": len(custom_paths),
            "internal_lexicon_used": use_internal,
            "interoperability_list_used": use_interop,
            "digital_transformation_list_used": use_digital,
        }
    }


def assess_uploaded_guideline_relevance(path: Optional[Path], sample_text: str = "") -> Dict[str, str]:
    if not path:
        return {"status": "not_uploaded", "message": "No additional guideline uploaded."}
    name = Path(path).name.lower()
    keywords = [
        "interoperability", "interoperable", "digital public service", "guideline", "framework",
        "eif", "eira", "assessment", "data exchange", "public administration", "διαλειτουργ",
    ]
    text = (name + " " + (sample_text or "")[:3000].lower())
    score = sum(1 for k in keywords if k in text)
    if score >= 2:
        status = "relevant"
        message = "The uploaded document appears relevant to interoperability assessment and will be considered as supplementary assessment-basis material."
    elif score == 1:
        status = "partially_relevant"
        message = "The uploaded document is partially relevant. It will be retained as supporting context, but not treated as a primary assessment basis unless confirmed by the assessor/user."
    else:
        status = "not_clearly_relevant"
        message = "The uploaded document is not clearly related to interoperability or digital public services. It will be stored as supplementary material but will not drive the assessment basis."
    return {"status": status, "message": message, "filename": Path(path).name, "keyword_score": str(score)}
