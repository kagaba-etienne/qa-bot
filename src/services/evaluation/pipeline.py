import csv
import json
from pathlib import Path

from src.services.evaluation.auto_cot import get_evaluation_steps
from src.services.evaluation.judge import score_metric
from src.services.evaluation.prompts import LLM_METRICS, REFUSAL_STRING

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
VARIANTS = (
    "baseline_no_rag",
    "baseline_with_rag",
    "finetuned_no_rag",
    "finetuned_with_rag",
)

QA_COLUMNS = (
    "question",
    "context",
    "gold_response",
    "model_response",
    "is_gold_refusal",
    "refusal_category",
    "citation_score",
    "refusal_exact_match",
    "false_refusal",
    "faithfulness_score",
    "relevance_score",
)
SUMMARY_COLUMNS = ("metric", "subset", "n", "mean")
SUMMARY_METRICS = (
    "citation_score",
    "refusal_exact_match",
    "false_refusal",
    "faithfulness_score",
    "relevance_score",
)
SUMMARY_SUBSETS = (
    "all",
    "positives",
    "gold_refusals",
    "refusal_A",
    "refusal_B",
    "refusal_C",
)
_REQUIRED_FIELDS = ("question", "context", "gold_response", "model_response")
_VALID_CATEGORIES = {"A", "B", "C", None}

PREDICTIONS_SCHEMA = """
predictions.jsonl — one JSON object per line:
  question: string
  context: string (the context the model saw; empty string if none)
  gold_response: string
  model_response: string
  refusal_category: "A", "B", "C", or null
""".strip()


def is_exact_refusal(text: str) -> bool:
    """True when the text is the mandated refusal, ignoring surrounding whitespace."""
    return text.strip() == REFUSAL_STRING


def refusal_fields(gold_response: str, model_response: str) -> dict:
    """Exact-match refusal labels. Blank cells are metrics that do not apply."""
    gold_is_refusal = is_exact_refusal(gold_response)
    model_is_refusal = is_exact_refusal(model_response)
    if gold_is_refusal:
        return {
            "is_gold_refusal": 1,
            "refusal_exact_match": 1 if model_is_refusal else 0,
            "false_refusal": "",
        }
    return {
        "is_gold_refusal": 0,
        "refusal_exact_match": "",
        "false_refusal": 1 if model_is_refusal else 0,
    }


def _cell(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _numeric(value):
    if value is None or value == "":
        return None
    return float(value)


def _is_gold_refusal_cell(value) -> bool:
    return str(value).strip() in {"1", "1.0"}


def row_key(question: str, model_response: str) -> tuple[str, str]:
    return (question, model_response)


def _in_subset(row: dict, subset: str) -> bool:
    gold_refusal = _is_gold_refusal_cell(row.get("is_gold_refusal", ""))
    category = str(row.get("refusal_category") or "")
    if subset == "all":
        return True
    if subset == "positives":
        return not gold_refusal
    if subset == "gold_refusals":
        return gold_refusal
    if subset.startswith("refusal_"):
        return gold_refusal and category == subset.split("_", 1)[1]
    return False


def summarize(rows: list[dict]) -> list[dict]:
    """Mean of each metric on each subset. Blank cells are ignored, and empty subsets are omitted."""
    summary = []
    for metric in SUMMARY_METRICS:
        for subset in SUMMARY_SUBSETS:
            values = [
                number
                for row in rows
                if _in_subset(row, subset)
                for number in [_numeric(row.get(metric))]
                if number is not None
            ]
            if not values:
                continue
            summary.append({
                "metric": metric,
                "subset": subset,
                "n": len(values),
                "mean": f"{(sum(values) / len(values)):.6f}",
            })
    return summary


def load_predictions(path: Path) -> list[dict]:
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
            for field in _REQUIRED_FIELDS:
                if field not in record:
                    raise ValueError(f"{path}:{line_no} is missing '{field}'.")
            normalized = {}
            for field in _REQUIRED_FIELDS:
                value = record[field]
                if field == "context" and value is None:
                    value = ""
                if not isinstance(value, str):
                    raise ValueError(
                        f"{path}:{line_no} field '{field}' must be a string.")
                normalized[field] = value
            category = record.get("refusal_category")
            if category not in _VALID_CATEGORIES:
                raise ValueError(
                    f"{path}:{line_no} field 'refusal_category' must be 'A', 'B', 'C', or null."
                )
            normalized["refusal_category"] = category
            rows.append(normalized)
    if not rows:
        raise ValueError(f"{path} has no predictions.")
    return rows


def _load_evaluated_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != QA_COLUMNS:
            raise ValueError(
                f"{path} columns do not match the expected evaluation schema."
            )
        return list(reader)


def _append_row(path: Path, row: dict) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=QA_COLUMNS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({column: _cell(row.get(column, ""))
                        for column in QA_COLUMNS})


def _write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summarize(rows))


def _score_prediction(prediction: dict, steps: dict[str, str], score_fn) -> dict:
    fields = refusal_fields(
        prediction["gold_response"], prediction["model_response"])
    row = {
        "question": prediction["question"],
        "context": prediction["context"],
        "gold_response": prediction["gold_response"],
        "model_response": prediction["model_response"],
        "refusal_category": prediction.get("refusal_category") or "",
        "citation_score": "",
        "faithfulness_score": "",
        "relevance_score": "",
        **fields,
    }
    metric_keys = ["faithfulness", "relevance"]
    if not fields["is_gold_refusal"]:
        metric_keys.insert(0, "citation")
    for metric_key in metric_keys:
        if metric_key not in steps or not str(steps[metric_key]).strip():
            raise ValueError(f"Missing Auto-CoT steps for {metric_key}.")
        score = score_fn(
            metric_key,
            steps[metric_key],
            prediction["question"],
            prediction["context"],
            prediction["model_response"],
        )
        row[f"{metric_key}_score"] = score
    return row


def evaluate_variant(results_dir: Path, steps: dict[str, str], score_fn=None) -> int:
    """Score one results folder. Returns how many predictions failed to score."""
    if score_fn is None:
        score_fn = score_metric
    predictions_path = results_dir / "predictions.jsonl"
    predictions = load_predictions(predictions_path)
    qa_path = results_dir / "evaluated_qa.csv"
    cached = {
        row_key(row["question"], row["model_response"]): row
        for row in _load_evaluated_rows(qa_path)
    }
    ordered = []
    failures = 0
    pending = sum(
        1 for prediction in predictions
        if row_key(prediction["question"], prediction["model_response"]) not in cached
    )
    print(
        f"Scoring {results_dir.name}: {len(predictions)} predictions, "
        f"{len(predictions) - pending} already scored."
    )
    for index, prediction in enumerate(predictions, start=1):
        key = row_key(prediction["question"], prediction["model_response"])
        if key in cached:
            ordered.append(cached[key])
            continue
        try:
            scored = _score_prediction(prediction, steps, score_fn)
        except Exception as exc:
            failures += 1
            print(f"  [{index}/{len(predictions)}] failed: {exc}")
            continue
        _append_row(qa_path, scored)
        cached[key] = {column: _cell(scored.get(column, ""))
                       for column in QA_COLUMNS}
        ordered.append(cached[key])
        print(f"  [{index}/{len(predictions)}] scored")
    _write_summary(results_dir / "summary.csv", ordered)
    return failures


def ensure_prediction_files(variants: list[str]) -> list[Path]:
    missing = []
    folders = []
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        folder = RESULTS_DIR / variant
        path = folder / "predictions.jsonl"
        folders.append(folder)
        if not path.is_file():
            missing.append(str(path))
    if missing:
        listed = "\n".join(missing)
        raise FileNotFoundError(
            f"Missing predictions file(s):\n{listed}\n\n{PREDICTIONS_SCHEMA}"
        )
    return folders


def run_evaluation(variants: list[str], refresh_cot: bool = False) -> int:
    """Score each variant. Prediction files are checked before any judge call."""
    folders = ensure_prediction_files(variants)
    steps = get_evaluation_steps(refresh=refresh_cot)
    for metric_key in LLM_METRICS:
        if metric_key not in steps:
            raise ValueError(
                f"Auto-CoT cache is missing steps for {metric_key}.")
    failures = 0
    for folder in folders:
        failures += evaluate_variant(folder, steps)
    return failures
