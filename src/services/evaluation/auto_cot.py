import json
from pathlib import Path

from src.services.evaluation.prompts import LLM_METRICS, build_cot_prompt

CACHE_PATH = Path(__file__).resolve().parent / "auto_cot.json"
JUDGE_MODEL = "gpt-4o"


def _clean_steps(text: str) -> str:
    cleaned = (text or "").strip()
    prefix = "Evaluation Steps:"
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix):].strip()
    if not cleaned:
        raise ValueError("Auto-CoT returned empty evaluation steps.")
    return cleaned


def _read_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{CACHE_PATH} is not valid JSON. Re-run with --refresh-cot."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"{CACHE_PATH} must be a JSON object. Re-run with --refresh-cot."
        )
    steps = {}
    for key, value in payload.items():
        if key not in LLM_METRICS or not isinstance(value, str) or not value.strip():
            continue
        steps[key] = value.strip()
    return steps


def _write_cache(steps: dict[str, str]) -> None:
    temporary = CACHE_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(steps, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(CACHE_PATH)


def _generate_steps(metric_key: str) -> str:
    from src.lib.openai import client

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": build_cot_prompt(metric_key)}],
    )
    return _clean_steps(response.choices[0].message.content)


def get_evaluation_steps(refresh: bool = False) -> dict[str, str]:
    """Return cached evaluation steps for every LLM metric, generating any that are missing."""
    steps = {} if refresh else _read_cache()
    missing = [key for key in LLM_METRICS if key not in steps]
    if not missing and not refresh:
        return steps

    keys_to_generate = list(LLM_METRICS) if refresh else missing
    for metric_key in keys_to_generate:
        print(f"Generating Auto-CoT steps for {metric_key}...")
        steps[metric_key] = _generate_steps(metric_key)
    _write_cache({key: steps[key] for key in LLM_METRICS})
    return {key: steps[key] for key in LLM_METRICS}
