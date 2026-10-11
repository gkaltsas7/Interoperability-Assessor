# Sensitive Terms Lexicon

The active internal lexicon is:

```text
lexicon/sensitive_terms_lexicon.xlsx
```

It is a human-readable Excel controlled vocabulary used by the assessor at runtime.

## Structure

The lexicon is organised around two main categories:

1. **Interoperability-sensitive terms**
   - Terms or phrases that may indicate closed systems, under-specified registries, proprietary formats, unclear data exchange, missing technical specifications or other interoperability risks.

2. **Digital-transformation-sensitive terms**
   - Terms or phrases that may indicate paper-based, physical, manual or legacy administrative processes, such as paper, physical presence, public books, newspaper publication, vague timing or unclear notification channels.

## Runtime behaviour

`utils/lexicon_engine.py` reads the Excel file, normalises the entries and detects matching terms in the assessed document.

A detected term is not treated as automatic non-compliance. It is a signal that the pipeline uses for contextual assessment, especially in the report section on sensitive terms, phrases and drafting patterns.

## Custom uploads

The tool can also read optional user-uploaded lexicons in:

```text
.xlsx
.csv
.txt
.json
```

Simple lists are accepted. Missing metadata are filled with safe defaults. Structured tables are also supported.

## Internal lexicon format

The official internal list is Excel-based. JSON support is retained for optional custom uploaded lists, not as a separate active internal lexicon.
