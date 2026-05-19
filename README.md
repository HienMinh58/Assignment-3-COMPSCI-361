# COMPSCI 361 Assignment 3

Collaborative machine learning project for classifying BBC news articles.

The project is organised so team members can work on separate Python modules first, test their changes, and then integrate the final work into a Jupyter notebook.

## Folder Structure

- `data/`: local datasets and generated data files.
- `notebooks/`: final and exploratory Jupyter notebooks.
- `src/`: reusable Python code for data loading, EDA, models, and evaluation.
- `tests/`: fast pytest tests for project code.
- `.github/workflows/`: GitHub Actions CI configuration.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
pytest -v
```

## Collaboration Workflow

1. Create a feature branch for your task.
2. Commit your changes with a clear message.
3. Open a Pull Request targeting `main`.
4. Wait for CI tests to pass before merging.

Suggested module ownership:

- Task 1 EDA: `src/eda.py`
- Naive Bayes: `src/nb_model.py`
- kNN: `src/knn_model.py`
- SVM: `src/svm_model.py`
- ANN: `src/ann_model.py`
- Model comparison: `src/evaluation.py`
