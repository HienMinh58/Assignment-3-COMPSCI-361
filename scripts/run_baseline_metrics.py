"""Run local Naive Bayes metric evaluation and print metrics as JSON."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_metrics import compute_baseline_metrics  # noqa: E402


TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"


def main() -> int:
    """Run baseline evaluation using local dataset files."""
    try:
        metrics = compute_baseline_metrics(TRAIN_PATH, TEST_PATH)
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
