# COMPSCI 361 Assignment 3

Collaborative machine learning project for classifying BBC news articles.

The project is organised so team members can work on separate Python modules first, test their changes, and then integrate the final work into a Jupyter notebook.

## Folder Structure

- `data/`: local datasets and generated data files.
- `notebooks/`: final and exploratory Jupyter notebooks.
- `src/`: reusable Python code for data loading, EDA, models, and evaluation.
- `src/classifiers/`: reusable classifier classes with a shared interface.
- `app/`: local FastAPI metrics service.
- `scripts/`: command-line helpers for local metric evaluation.
- `tests/`: fast pytest tests for project code.
- `.github/workflows/`: GitHub Actions CI configuration.

## Setup

Clone the repository:

```bash
git clone https://github.com/HienMinh58/Assignment-3-COMPSCI-361.git
cd Assignment-3-COMPSCI-361
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
pytest -v
```

## Local Dataset Setup

Place the assignment datasets on your own machine at:

```text
data/train.csv
data/test.csv
```

These CSV files are intentionally not committed to GitHub. The API, Docker service, and local metrics script read from these local files only.

## Classifier Architecture

Each assignment model has its own classifier module in `src/classifiers/`:

- `src/classifiers/nb_classifier.py`
- `src/classifiers/knn_classifier.py`
- `src/classifiers/svm_classifier.py`
- `src/classifiers/ann_classifier.py`

All classifier classes expose the same simple interface:

- `fit(train_texts, train_labels)`
- `predict(test_texts)`
- `evaluate(test_texts, test_labels)`

`src/classifiers/nb_classifier.py` is the reference implementation. It uses `TfidfVectorizer` and `MultinomialNB`, while kNN, SVM, and ANN are placeholders for teammates to complete later. Classifiers are created through `src/classifier_factory.py`, and local metric evaluation for every endpoint is coordinated by `src/metrics_service.py`.

The older top-level model placeholders, such as `src/nb_model.py` and `src/ann_model.py`, have been removed to avoid confusion. New model work should go in `src/classifiers/`.

## Run Classifier Metrics Script

Evaluate all four classifiers:

```bash
python scripts/run_classifier_metrics.py
```

The script prints one combined JSON object. Implemented classifiers include their metrics, while unfinished classifiers return a `not_implemented` status without crashing the script.

The older baseline helper is still available:

```bash
python scripts/run_baseline_metrics.py
```

## Run FastAPI Metrics Service Locally

```bash
uvicorn app.main:app --reload
```

The `--reload` flag restarts the API automatically when files under `app/` or `src/` change.

Endpoints:

- `GET /health`
- `GET /metrics/nb`
- `GET /metrics/knn`
- `GET /metrics/svm`
- `GET /metrics/ann`

Only Naive Bayes is implemented right now. The kNN, SVM, and ANN endpoints are already wired into the metrics service, but they return HTTP 501 until teammates implement the matching classifier classes. Once a classifier is implemented, the same endpoint can be used to test its local metrics.

Interactive API docs:

- `http://localhost:8000/docs`

## Run with Docker

```bash
docker compose up --build
```

Docker Compose runs uvicorn with auto-reload and mounts the local source folders:

```text
./app:/app/app
./src:/app/src
./scripts:/app/scripts
./data:/app/data
```

That means changes to FastAPI routes or classifier files are picked up automatically by the running container. If you change `requirements.txt`, rebuild with `docker compose up --build`.

Open:

- `http://localhost:8000/health`
- `http://localhost:8000/metrics/nb`
- `http://localhost:8000/docs`

Stop the service with:

```bash
docker compose down
```

## Collaboration Workflow

1. Create a feature branch for your task.
2. Commit your changes with a clear message.
3. Open a Pull Request targeting `main`.
4. Wait for CI tests to pass before merging.

Suggested module ownership:

- Task 1 EDA: `src/eda.py`
- Naive Bayes reference implementation: `src/classifiers/nb_classifier.py`
- kNN placeholder: `src/classifiers/knn_classifier.py`
- SVM placeholder: `src/classifiers/svm_classifier.py`
- ANN placeholder: `src/classifiers/ann_classifier.py`
- Model comparison: `src/evaluation.py`
