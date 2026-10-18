import json
from pathlib import Path

from src.config.env import PROCESSED_DIR
from src.lib.qwen import (
    BASE_MODEL,
    FINETUNED_MODEL,
    SYSTEM_PROMPT,
    build_messages,
    qwen,
)
from src.services.evaluation.pipeline import RESULTS_DIR, VARIANTS
from src.services.evaluation.prompts import REFUSAL_STRING

VAL_PATH = PROCESSED_DIR / "qlora_val.jsonl"
_VALID_CATEGORIES = {"A", "B", "C", None}
_VARIANT_MODELS = {
    "baseline_no_rag": BASE_MODEL,
    "baseline_with_rag": BASE_MODEL,
    "finetuned_no_rag": FINETUNED_MODEL,
    "finetuned_with_rag": FINETUNED_MODEL,
}


def _uses_stored_context(variant: str) -> bool:
    return variant.endswith("_with_rag")


def load_validation_rows(path: Path = VAL_PATH) -> list[dict]:
    """Load the QLoRA validation set used to generate evaluation predictions."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation file: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no} is not valid JSON.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object.")
            for field in ("question", "context", "response"):
                if not isinstance(record.get(field), str):
                    raise ValueError(
                        f"{path}:{line_no} field '{field}' must be a string.")
            category = record.get("_refusal_category")
            if category not in _VALID_CATEGORIES:
                raise ValueError(
                    f"{path}:{line_no} field '_refusal_category' must be 'A', 'B', 'C', or null."
                )
            rows.append(record)
    if not rows:
        raise ValueError(f"{path} has no validation rows.")
    return rows


def _load_completed_questions(path: Path) -> set[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no} is not valid JSON.") from exc
            question = record.get("question") if isinstance(
                record, dict) else None
            if not isinstance(question, str) or not question:
                raise ValueError(f"{path}:{line_no} is missing 'question'.")
            done.add(question)
    return done


def _append_prediction(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_variant(variant: str, rows: list[dict]) -> int:
    """Write one variant's predictions.jsonl. Returns how many rows failed to generate."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    path = RESULTS_DIR / variant / "predictions.jsonl"
    done = _load_completed_questions(path)
    pending = sum(1 for row in rows if row["question"] not in done)
    print(
        f"Generating {variant}: {len(rows)} validation rows, "
        f"{len(rows) - pending} already saved."
    )
    failures = 0
    model = _VARIANT_MODELS[variant]
    use_context = _uses_stored_context(variant)
    for index, row in enumerate(rows, start=1):
        if row["question"] in done:
            continue
        context = row["context"] if use_context else ""
        messages = build_messages(
            row["question"],
            context,
            row.get("instruction") or SYSTEM_PROMPT,
        )
        try:
            model_response = qwen(messages, model=model)
        except ValueError:
            raise
        except Exception as exc:
            failures += 1
            print(f"  [{index}/{len(rows)}] failed: {exc}")
            continue
        seen_context = context.strip() if isinstance(context, str) else ""
        _append_prediction(path, {
            "question": row["question"],
            "context": seen_context,
            "gold_response": row["response"] if use_context else REFUSAL_STRING,
            "model_response": model_response,
            "refusal_category": row.get("_refusal_category"),
        })
        done.add(row["question"])
        print(f"  [{index}/{len(rows)}] saved")
    return failures


def generate_predictions(variants: list[str]) -> int:
    """Generate predictions for each variant. Returns how many rows failed."""
    rows = load_validation_rows()
    failures = 0
    for variant in variants:
        failures += generate_variant(variant, rows)
    return failures
