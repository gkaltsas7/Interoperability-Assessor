import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing. Add it to your local .env file.")

client = OpenAI(api_key=api_key)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1")


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _call_openai(input_text: str, max_output_tokens: int = 6000) -> str:
    response = client.responses.create(
        model=MODEL_NAME,
        input=input_text,
        max_output_tokens=max_output_tokens,
    )
    return response.output_text.strip()


def generate_text(system_prompt: str, task_prompt: str, max_output_tokens: int = 6000) -> str:
    full_input = f"""
{system_prompt}

TASK:
{task_prompt}
"""
    return _call_openai(full_input, max_output_tokens=max_output_tokens)


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def generate_json(system_prompt: str, task_prompt: str, max_output_tokens: int = 6000) -> Dict[str, Any]:
    json_rules = """
STRICT JSON OUTPUT RULES:
- Return ONLY valid JSON.
- Do NOT include markdown fences.
- Do NOT include explanatory text outside JSON.
- Use double quotes for all keys and string values.
- If evidence is missing, use "Missing Information" and set confidence to "low".
"""
    raw = generate_text(
        system_prompt=system_prompt,
        task_prompt=f"{json_rules}\n\n{task_prompt}",
        max_output_tokens=max_output_tokens,
    )
    try:
        return _extract_json(raw)
    except Exception:
        repair_prompt = f"""
Convert the following invalid output into valid JSON only.
Do not add new facts.

INVALID OUTPUT:
{raw}
"""
        repaired = generate_text(system_prompt=json_rules, task_prompt=repair_prompt, max_output_tokens=max_output_tokens)
        return _extract_json(repaired)
