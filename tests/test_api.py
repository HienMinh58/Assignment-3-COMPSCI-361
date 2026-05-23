from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app import main


def _write_dataset(path: Path) -> None:
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4],
            "Article": [
                "team wins football match",
                "government election policy",
                "football team scores goal",
                "minister announces policy",
            ],
            "Category": ["sport", "politics", "sport", "politics"],
        }
    )
    df.to_csv(path, index=False)


def test_health_returns_ok():
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_nb_metrics_returns_results_for_temporary_datasets(tmp_path, monkeypatch):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_dataset(train_path)
    _write_dataset(test_path)
    monkeypatch.setattr(main, "TRAIN_PATH", train_path)
    monkeypatch.setattr(main, "TEST_PATH", test_path)
    client = TestClient(main.app)

    response = client.get("/metrics/nb")

    assert response.status_code == 200
    result = response.json()
    assert result["model"] == "nb"
    assert "dataset" in result
    assert set(result["metrics"]) == {"accuracy", "f1_macro", "f1_weighted"}


def test_svm_metrics_returns_results_for_temporary_datasets(tmp_path, monkeypatch):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_dataset(train_path)
    _write_dataset(test_path)
    monkeypatch.setattr(main, "TRAIN_PATH", train_path)
    monkeypatch.setattr(main, "TEST_PATH", test_path)
    client = TestClient(main.app)

    response = client.get("/metrics/svm")

    assert response.status_code == 200
    result = response.json()
    assert result["model"] == "svm"
    assert "dataset" in result
    assert set(result["metrics"]) == {"accuracy", "f1_macro", "f1_weighted"}


def test_knn_ann_metric_endpoints_placeholder_or_valid_metrics(tmp_path, monkeypatch):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_dataset(train_path)
    _write_dataset(test_path)
    monkeypatch.setattr(main, "TRAIN_PATH", train_path)
    monkeypatch.setattr(main, "TEST_PATH", test_path)
    client = TestClient(main.app)

    for model_name in ["knn", "ann"]:
        path = f"/metrics/{model_name}"
        response = client.get(path)
        result = response.json()

        if response.status_code == 501:
            assert "not been implemented yet" in result["detail"]
        else:
            assert response.status_code == 200
            assert result["model"] == model_name
            assert "dataset" in result
            assert set(result["metrics"]) == {"accuracy", "f1_macro", "f1_weighted"}
