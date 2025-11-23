# Project 05:  Ensemble ML, Spiral (Wine)

**Author:**  James Pinkston<br>
**Date:**  November 23, 2025

## Project Overview

In this project, we learn how to implement and evaluate more complex models when simpler techniques aren't enough. We'll explore ensemble models, a powerful approach in machine learning that combines multiple models to improve performance. Ensemble methods often outperform individual models by reducing overfitting and improving generalization.

## Activate Project Virtual Environment

Since this project (and future projects for this course) all use the same repository, they will all use the same local virtual environment created for Project 01. All that needs to be done for each project is to activate.

```shell
.\.venv\Scripts\activate
```

## Create new project directory and files with VS Code

Inside the notebooks directory, create a project05 subdirectory with the following files inside:
ensemble_pinkston.ipynb
README.md

## Run the following test and prepatory commands

```shell
git add .
uvx ruff check --fix
uvx pre-commit autoupdate
uv run pre-commit run --all-files
git add .
uv run pytest
```

## Run initial Git Add / Commit / Push

```shell
git add .
git commit -m "initial commit"
git push -u origin main
```

### Run the Jupyter Notebook

notebooks/project05/ensemble_pinkston.ipynb

## Project Details

### Dataset

- Source: <a href="https://archive.ics.uci.edu/ml/datasets/Wine+Quality" target="_blank">Wine Quality Dataset</a>
- Key Features: fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality, color
- Target Variable: quality_numeric

### Data Preparation and Feature Engineering

- Assign the quality data a string based on the quality score (q)
-- low: q <=4
-- medium: q <= 6
-- high: q > 6
- Split the quality column into two new columns based on previous results
-- quality_label: low, medium, high
-- quality_numeric: low = 0, medium = 1, high = 2

### Feature Selection and Justification

- Drop all columns except quality and the newly created quality_label and quality_numeric columns
- Features (X) = quality, quality_label, quality_numeric
- Target (y) = quality_numeric

### Ensemble Models

1. Random Forest
2. MLP Classifier

## Evaluation

### Summary of All Models

| Model               | Train Accuracy | Test Accuracy | Train F1   | Test F1   | Accuracy Gap | F1 Gap    |
|--------------------|----------------|---------------|------------|-----------|--------------|-----------|
| Random Forest (100)| 1.0000         | 0.8875        | 1.0000     | 0.8661    | 0.1125       | 0.1339    |
| MLP Classifier     | 0.8514         | 0.8438        | 0.8141     | 0.8073    | 0.0077       | 0.0068    |
  

## Conclusions and Insights

1. Project Summary:  <span style="color:darkgreen;">**In this analysis, we evaluated two machine learning models — Random Forest and Multi-Layer Perceptron (MLP) classifiers — to predict red wine quality. We measured model performance using both accuracy and weighted F1 score on the training and test datasets, and calculated gaps to assess overfitting.**</span>

2. Performance Trends:  <span style="color:darkgreen;">**The Random Forest classifier achieved the highest test accuracy at 88.75%, but also showed a notable training/test gap (11.25%), indicating some overfitting. In contrast, the MLP classifier had slightly lower test accuracy (84.38%) but a much smaller accuracy gap (0.76%), suggesting more stable generalization.**</span>

3. Insights on Why Models Behave Differently:  <span style="color:darkgreen;">**Tree-based models, like Random Forest, perform well on this dataset because they capture non-linear relationships and interactions between wine chemical properties, but they can overfit when too many trees or deep splits are used. Neural networks, such as MLP, tend to generalize more consistently and handle complex patterns, though they may require careful tuning and more data to reach peak accuracy.**</span>

4. Next Steps:  <span style="color:darkgreen;">**While this project focused on using 2 of the 9 models, I think when working on a professional team, it would be prudent to use as many models as possible, and then compare them together as a team. The team could then decide on which model's (models') results to use when presenting their findings to the project shareholders.**</span>