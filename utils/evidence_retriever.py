import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class EvidenceChunk:
    chunk_id: str
    kind: str
    reference: str
    text: str
    score: int = 0


GENERAL_KEYWORDS = [
    # Core IEA / EIF indicators
    "interoperability", "cross-border", "digital", "electronic", "public service",
    "network and information system", "data exchange", "database", "registry", "register",
    "platform", "information system", "machine-readable", "metadata", "api", "interface",
    "authentication", "authorisation", "authorization", "logging", "audit", "record-keeping",
    "technical documentation", "transparency", "governance", "authority", "provider", "deployer",
    "public authority", "Union institution", "Member State", "market surveillance", "conformity assessment",
    # Broad assessor-discovery indicators from the professor's examples/template
    "electronic platform", "special electronic platform", "wills registry", "μητρώο", "πλατφόρμα",
    "publication", "δημοσίευση", "public book", "δημόσιο βιβλίο", "certificate", "πιστοποιητικό",
    "consular", "προξενική", "abroad", "εξωτερικό", "foreign", "public documents",
    "submission", "transmission", "notification", "send a copy", "αντίγραφο", "διαβίβαση", "αποστολή",
    "by any suitable means", "without culpable delay", "without undue delay", "press", "newspaper", "εφημερίδα",
]

STAGE_PROFILES: Dict[str, Dict[str, Any]] = {
    "discovery": {
        "keywords": [
            "electronic platform", "special electronic platform", "registry", "database", "wills registry", "public book",
            "publication", "publish", "certificate", "protocol", "notary", "notarial", "consular", "abroad",
            "foreign", "public documents", "submission", "transmission", "notification", "copy", "press", "newspaper",
            "without delay", "without undue delay", "without culpable delay", "by any suitable means", "suitable means",
            "electronic signature", "seal", "timestamp", "archive", "retention", "data model", "metadata",
            "access rights", "authentication", "audit", "governance", "implementing decision", "procurement",
            "Άρθρο 1752", "Άρθρο 1759", "Άρθρο 1760", "Άρθρο 1761", "Άρθρο 1896", "Άρθρο 1897",
            "Άρθρο 1923", "Άρθρο 1925", "Άρθρο 808Α", "Άρθρο 808Β",
        ],
        "prefer": ["article", "page"],
    },
    "profile": {
        "keywords": ["title", "purpose", "objective", "scope", "Article 1", "Article 2", "subject matter"],
        "prefer": ["article", "page"],
    },
    "binding_requirements": {
        "keywords": [
            "shall", "must", "prohibited", "requirement", "obligation", "providers shall", "deployers shall",
            "Article 5", "Article 6", "Article 9", "Article 11", "Article 12", "Article 13", "Article 14",
            "Article 15", "Article 16", "Article 26", "Article 27", "Article 49", "Article 50", "Annex III",
            "high-risk", "EU database", "registration", "technical documentation", "logging",
            "record-keeping", "transparency", "fundamental rights impact assessment",
        ],
        "prefer": ["article", "annex"],
    },
    "trigger": {
        "keywords": [
            "binding requirements", "public sector", "Union entity", "digital public service", "cross-border",
            "network and information systems", "Article 3", "decision", "adopted", "implementation",
            "procurement", "shall", "Union institutions", "public authorities",
        ],
        "prefer": ["article", "page"],
    },
    "services": {
        "keywords": [
            "public services", "Union institutions", "public authorities", "deployer", "provider", "AI systems used by",
            "EU database", "registration", "notification", "market surveillance", "competent authorities",
            "data exchange", "reporting", "cross-border", "database", "platform",
        ],
        "prefer": ["article", "annex"],
    },
    "layers": {
        "keywords": [
            "legal", "organisational", "semantic", "technical", "governance", "EU database", "registration",
            "technical documentation", "logging", "record-keeping", "instructions for use", "transparency",
            "human oversight", "accuracy", "robustness", "cybersecurity", "market surveillance", "notifying authority",
            "AI Office", "Board", "competent authorities", "common specifications", "harmonised standards",
        ],
        "prefer": ["article", "annex"],
    },
    "technical": {
        "keywords": [
            "technical documentation", "logging", "record-keeping", "EU database", "registration", "database",
            "machine-readable", "interface", "api", "instructions for use", "accuracy", "robustness",
            "cybersecurity", "data governance", "data sets", "metadata", "common specifications",
            "harmonised standards", "Annex IV", "Annex VIII", "Article 11", "Article 12", "Article 49",
        ],
        "prefer": ["article", "annex"],
    },
    "governance": {
        "keywords": [
            "governance", "AI Office", "Board", "competent authority", "notifying authority", "market surveillance",
            "Commission", "Member States", "cooperation", "supervision", "monitoring", "reporting",
            "Article 64", "Article 65", "Article 70", "Article 74", "Article 75", "Article 77", "Article 78",
        ],
        "prefer": ["article"],
    },
    "solutions": {
        "keywords": [
            "harmonised standards", "common specifications", "standardisation", "codes of conduct", "EU database",
            "interoperability solutions", "technical specifications", "machine-readable", "electronic", "metadata",
        ],
        "prefer": ["article", "annex"],
    },
    "recommendations": {
        "keywords": [
            "missing", "unclear", "shall", "database", "technical documentation", "logging", "governance",
            "registration", "market surveillance", "standard", "common specifications", "implementation",
        ],
        "prefer": ["article", "annex"],
    },
}


def _normalise(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _extract_reference(text: str, fallback: str) -> str:
    first = text.splitlines()[0] if text.splitlines() else ""
    page_match = re.search(r"\[PAGE\s+(\d+)\]", first, flags=re.I)
    if page_match:
        return f"PAGE {page_match.group(1)}"
    art_match = re.search(r"\b(Article|ARTICLE|Άρθρο|ΑΡΘΡΟ)\s+(\d+[A-Za-zΑ-Ωα-ω]?)", text)
    if art_match:
        return f"{art_match.group(1).title()} {art_match.group(2)}"
    annex_match = re.search(r"\b(Annex|ANNEX)\s+([IVXLCDM]+|[A-Z]|\d+)", text)
    if annex_match:
        return f"Annex {annex_match.group(2)}"
    return fallback


def split_document(document_text: str) -> List[EvidenceChunk]:
    text = document_text.strip()
    if not text:
        return []

    page_units: List[str] = []
    if "[PAGE " in text:
        current: List[str] = []
        for line in text.splitlines():
            if line.startswith("[PAGE ") and current:
                page_units.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            page_units.append("\n".join(current))
    else:
        page_units = [text]

    chunks: List[EvidenceChunk] = []
    for page_index, page in enumerate(page_units, start=1):
        page = _normalise(page)
        if not page:
            continue

        # EU/national legal acts often have Article/Annex headings inside pages; split while keeping fallback page chunk.
        parts = re.split(r"(?=\b(?:Article|ARTICLE|Άρθρο|ΑΡΘΡΟ)\s+\d+[A-Za-zΑ-Ωα-ω]?\b|\b(?:Annex|ANNEX)\s+(?:[IVXLCDM]+|[A-Z]|\d+)\b)", page)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) > 1:
            for part_index, part in enumerate(parts, start=1):
                kind = "annex" if re.search(r"\b(?:Annex|ANNEX)\b", part) else "article" if re.search(r"\b(?:Article|ARTICLE|Άρθρο|ΑΡΘΡΟ)\s+\d+", part) else "page"
                chunks.append(EvidenceChunk(
                    chunk_id=f"P{page_index}-C{part_index}",
                    kind=kind,
                    reference=_extract_reference(part, f"PAGE {page_index}"),
                    text=part,
                ))
        else:
            chunks.append(EvidenceChunk(
                chunk_id=f"P{page_index}",
                kind="page",
                reference=_extract_reference(page, f"PAGE {page_index}"),
                text=page,
            ))

    return chunks


def score_chunk(chunk: EvidenceChunk, stage: str) -> int:
    low = chunk.text.lower()
    profile = STAGE_PROFILES.get(stage, STAGE_PROFILES.get("binding_requirements", {}))
    keywords = profile.get("keywords", [])
    prefer = profile.get("prefer", [])

    score = 0
    for kw in GENERAL_KEYWORDS:
        if kw.lower() in low:
            score += 2
    for kw in keywords:
        if kw.lower() in low:
            score += 7
    if chunk.kind in prefer:
        score += 8
    if re.search(r"\bArticle\s+\d+", chunk.text, flags=re.I):
        score += 5
    if re.search(r"\bAnnex\b", chunk.text, flags=re.I):
        score += 4
    if 800 <= len(chunk.text) <= 14000:
        score += 3
    if len(chunk.text) > 25000:
        score -= 4
    return score


def retrieve_evidence(
    document_text: str,
    stage: str,
    top_k: int = 18,
    max_chars: int = 130000,
    include_front_matter: bool = True,
) -> Dict[str, Any]:
    chunks = split_document(document_text)
    scored: List[Tuple[int, EvidenceChunk]] = []
    for chunk in chunks:
        scored.append((score_chunk(chunk, stage), chunk))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[EvidenceChunk] = []
    selected_ids = set()

    if include_front_matter:
        for chunk in chunks[:2]:
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)

    for score, chunk in scored:
        if score <= 0 or chunk.chunk_id in selected_ids:
            continue
        selected.append(chunk)
        selected_ids.add(chunk.chunk_id)
        if len(selected) >= top_k:
            break

    corpus_parts: List[str] = []
    total = 0
    for chunk in selected:
        block = f"[{chunk.chunk_id} | {chunk.kind.upper()} | {chunk.reference}]\n{chunk.text}"
        if total + len(block) > max_chars:
            continue
        corpus_parts.append(block)
        total += len(block)

    all_refs = [c.reference for c in selected]
    return {
        "stage": stage,
        "corpus": "\n\n".join(corpus_parts),
        "diagnostics": {
            "total_chunks_scanned": len(chunks),
            "selected_chunks": len(corpus_parts),
            "selected_characters": total,
            "selected_references": all_refs[:40],
            "top_scored_references": [c.reference for _, c in scored[:20]],
        },
    }


def retrieve_all_stage_evidence(document_text: str) -> Dict[str, Dict[str, Any]]:
    return {
        # Broad discovery is deliberately generous. It prevents the agent from locking onto one article/flow too early.
        "discovery": retrieve_evidence(document_text, "discovery", top_k=50, max_chars=230000),
        "profile": retrieve_evidence(document_text, "profile", top_k=14, max_chars=90000),
        "binding_requirements": retrieve_evidence(document_text, "binding_requirements", top_k=28, max_chars=160000),
        "trigger": retrieve_evidence(document_text, "trigger", top_k=18, max_chars=120000),
        "services": retrieve_evidence(document_text, "services", top_k=20, max_chars=130000),
        "layers": retrieve_evidence(document_text, "layers", top_k=28, max_chars=170000),
        "technical": retrieve_evidence(document_text, "technical", top_k=24, max_chars=150000),
        "governance": retrieve_evidence(document_text, "governance", top_k=22, max_chars=140000),
        "solutions": retrieve_evidence(document_text, "solutions", top_k=20, max_chars=130000),
        "recommendations": retrieve_evidence(document_text, "recommendations", top_k=20, max_chars=130000),
    }
