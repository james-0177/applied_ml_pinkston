# Project 04:  Predicting a Continuous Target with Regression (Titanic)

**Author:**  James Pinkston<br>
**Date:**  November 16, 2025

## Project Overview

In this project, we’ll predict fare, the amount of money paid for the journey, using features in the Titanic dataset.

## Activate Project Virtual Environment

Since this project (and future projects for this course) all use the same repository, they will all use the same local virtual environment created for Project 01. All that needs to be done for each project is to activate.

```shell
.\.venv\Scripts\activate
```

## Create new project directory and files with VS Code

Inside the notebooks directory, create a project04 subdirectory with the following files inside:
ml04_pinkston.ipynb
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

notebooks/project04/ml04_pinkston.ipynb

## Project Details

### Dataset

- Source: Titanic dataset within Seaborn
- Key Features: survived, pclass, sex, age, sibsp, parch, fare, embarked, alone
- Target Variable: survived

### Data Exploration

- Missing values imputed:
  - age - median
- Drop rows with missing data
- Categorical variables encoded numerically
  - sex:  male = 0, female = 1
  - alone

### Feature Engineering

- Create Numeric Feature:  family_size = sibsp + parch + 1

### Feature Selection and Justification

- Case 1. age
- Case 2. family_size
- Case 3. age + family_size
- Case 4. sex + alone

### Regression Models

1. Linear Regression
2. Ridge Regression
3. Elastic Net
4. Polynomial Regression

### Evaluation

Linear R²: 0.191  
Linear RMSE: 34.21  
Linear MAE: 21.97  

Ridge R²: 0.191  
Ridge RMSE: 34.21  
Ridge MAE: 21.97  

ElasticNet R²: 0.155  
ElasticNet RMSE: 34.96  
ElasticNet MAE: 22.63  

Polynomial R²: -0.003  
Polynomial RMSE: 38.10  
Polynomial MAE: 25.30  

## Final Thoughts & Insights

### Summarize Findings

1. What features were most useful?  <span style="color:darkgreen;">**Case 4 (sex and alone) were the most useful features that I tested.**</span>

2. What regression model performed best?  <span style="color:darkgreen;">**Both the Linear Regression Model and the Ridge Model performed the best as they had the same results.**</span>

3. How did model complexity or regularization affect results?  <span style="color:darkgreen;">**Model complexity (using age) did not perform better, but that was mainly due to age being a nearly useless predictor.**</span>

### Challenges

1. Was fare hard to predict? Why?  <span style="color:darkgreen;">**Yes, mostly due to the features that were selected. Age was nearly useless, and quite possibly the worst predictor. Pclass would have been much better, but was not chosen, even for the custom Case 4.**</span>

2. Did skew or outliers impact the models?  <span style="color:darkgreen;">**Yes. A few passengers paid very high fares. Outliers can make R^2 and RMSE misleading.**</span>

### Optional Next Steps

1. Try different features besides the ones used (e.g., pclass, sex if you didn't use them this time):  <span style="color:darkgreen;">**Pclass would probably be the best feature to use for a fare prediction model.**</span>

2. Try predicting age instead of fare:  <span style="color:darkgreen;">**Age would have a much different correlation structure, features useful would probably be sex, pclass, and family_size.**</span>

3. Explore log transformation of fare to reduce skew:  <span style="color:darkgreen;">**This would probably greatly help the prediction model as it compresses large values, reducing the difference between extreme and typical fares.**</span>