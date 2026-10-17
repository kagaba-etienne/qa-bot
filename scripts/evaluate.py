from src.services.evaluation.pipeline import VARIANTS, run_evaluation
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score saved model predictions with G-Eval."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--variant",
        choices=VARIANTS,
        help="Results folder under data/results/ to score.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="all_variants",
        help="Score baseline_no_rag, finetuned_no_rag, and finetuned_with_rag.",
    )
    parser.add_argument(
        "--refresh-cot",
        action="store_true",
        help="Regenerate the cached Auto-CoT evaluation steps.",
    )
    args = parser.parse_args()
    variants = list(VARIANTS) if args.all_variants else [args.variant]
    try:
        failures = run_evaluation(variants, refresh_cot=args.refresh_cot)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
