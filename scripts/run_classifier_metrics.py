"""Run local classifier metric evaluation and print metrics as JSON."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics_service import (  # noqa: E402
    evaluate_ann,
    evaluate_knn,
    evaluate_nb,
    evaluate_svm,
)


TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"


EVALUATORS = {
    "nb": evaluate_nb,
    "knn": evaluate_knn,
    "svm": evaluate_svm,
    "ann": evaluate_ann,
}


def evaluate_all_models() -> dict:
    """Evaluate all classifiers and report placeholders without crashing."""
    results = {}
    for model_name, evaluator in EVALUATORS.items():
        try:
            results[model_name] = evaluator(TRAIN_PATH, TEST_PATH)
        except NotImplementedError as exc:
            results[model_name] = {
                "status": "not_implemented",
                "detail": str(exc),
            }
    return {"models": results}


def main() -> int:
    """Run local metrics for all classifier endpoints."""
    try:
        metrics = evaluate_all_models()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Place the assignment datasets at data/train.csv and data/test.csv.",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
