"""Local FastAPI service for classifier metrics."""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.metrics_service import evaluate_ann, evaluate_knn, evaluate_nb, evaluate_svm


TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")

app = FastAPI(title="COMPSCI 361 Assignment 3 Metrics API")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple service health check."""
    return {"status": "ok"}


def _handle_metrics_error(exc: Exception) -> HTTPException:
    """Convert expected local metrics errors into HTTP responses."""
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, pd.errors.ParserError):
        return HTTPException(
            status_code=400,
            detail=f"Could not parse dataset CSV file: {exc}",
        )
    if isinstance(exc, NotImplementedError):
        return HTTPException(status_code=501, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))

    return HTTPException(status_code=500, detail="Unexpected metrics service error.")


@app.get("/metrics/nb")
def nb_metrics() -> dict:
    """Evaluate Naive Bayes using local train and test CSV files."""
    try:
        return evaluate_nb(TRAIN_PATH, TEST_PATH)
    except (FileNotFoundError, pd.errors.ParserError, ValueError, NotImplementedError) as exc:
        raise _handle_metrics_error(exc) from exc


@app.get("/metrics/knn")
def knn_metrics() -> dict:
    """Evaluate kNN using local train and test CSV files."""
    try:
        return evaluate_knn(TRAIN_PATH, TEST_PATH)
    except (FileNotFoundError, pd.errors.ParserError, ValueError, NotImplementedError) as exc:
        raise _handle_metrics_error(exc) from exc


@app.get("/metrics/svm")
def svm_metrics() -> dict:
    """Evaluate SVM using local train and test CSV files."""
    try:
        return evaluate_svm(TRAIN_PATH, TEST_PATH)
    except (FileNotFoundError, pd.errors.ParserError, ValueError, NotImplementedError) as exc:
        raise _handle_metrics_error(exc) from exc


@app.get("/metrics/ann")
def ann_metrics() -> dict:
    """Evaluate ANN using local train and test CSV files."""
    try:
        return evaluate_ann(TRAIN_PATH, TEST_PATH)
    except (FileNotFoundError, pd.errors.ParserError, ValueError, NotImplementedError) as exc:
        raise _handle_metrics_error(exc) from exc
