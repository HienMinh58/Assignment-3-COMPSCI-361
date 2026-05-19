from scripts import run_classifier_metrics


def test_evaluate_all_models_includes_metrics_and_not_implemented(monkeypatch):
    def fake_nb(train_path, test_path):
        return {
            "model": "nb",
            "dataset": {"train_rows": 2, "test_rows": 1},
            "metrics": {
                "accuracy": 1.0,
                "f1_macro": 1.0,
                "f1_weighted": 1.0,
            },
        }

    def not_implemented(train_path, test_path):
        raise NotImplementedError("This classifier has not been implemented yet.")

    monkeypatch.setattr(
        run_classifier_metrics,
        "EVALUATORS",
        {
            "nb": fake_nb,
            "knn": not_implemented,
            "svm": not_implemented,
            "ann": not_implemented,
        },
    )

    result = run_classifier_metrics.evaluate_all_models()

    assert result["models"]["nb"]["model"] == "nb"
    assert result["models"]["knn"] == {
        "status": "not_implemented",
        "detail": "This classifier has not been implemented yet.",
    }
    assert result["models"]["svm"]["status"] == "not_implemented"
    assert result["models"]["ann"]["status"] == "not_implemented"


def test_evaluate_all_models_does_not_catch_file_errors(monkeypatch):
    def missing_file(train_path, test_path):
        raise FileNotFoundError("missing train.csv")

    monkeypatch.setattr(
        run_classifier_metrics,
        "EVALUATORS",
        {"nb": missing_file},
    )

    try:
        run_classifier_metrics.evaluate_all_models()
    except FileNotFoundError as exc:
        assert "missing train.csv" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
