import argparse
import sys

from src.services.evaluation.pipeline import VARIANTS, run_evaluation
from src.services.evaluation.predict import generate_predictions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate validation predictions with the RunPod endpoint and score them."
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        help="Results folder under data/results/ to generate. Defaults to every variant.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Write predictions.jsonl only and skip G-Eval.",
    )
    args = parser.parse_args()
    variants = [args.variant] if args.variant else list(VARIANTS)
    try:
        failures = generate_predictions(variants)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if failures:
        print(
            f"{failures} prediction(s) failed. Re-run to resume; scoring was skipped.",
            file=sys.stderr,
        )
        return 1
    if args.no_score:
        return 0
    try:
        score_failures = run_evaluation(variants)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 1 if score_failures else 0


if __name__ == "__main__":
    sys.exit(main())
