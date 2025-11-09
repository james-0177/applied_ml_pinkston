# Project 03:  Building a Classifier (Titanic)

## Project Overview

In this project, we use the Titanic dataset to build and compare three classifiers:  Decision Tree, Support Vector Machine, and Neural Network.

We test each model across three different feature sets and evaluate their performance in predicting passenger survival.

Decision Tree (DT) – Splits data into smaller groups based on simple rules.

Support Vector Machine (SVM) – Finds the best boundary (hyperplane) that separates data into classes.

Neural Network (NN) – Uses layers of connected “neurons” to learn complex, non-linear patterns.

## Activate Project Virtual Environment

Since this project (and future projects for this course) all use the same repository, they will all use the same local virtual environment created for Project 01. All that needs to be done for each project is to activate.

```shell
.\.venv\Scripts\activate
```

## Create new project directory and files with VS Code

Inside the notebooks directory, create a project03 subdirectory with the following files inside:
ml03_pinkston.ipynb
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

## Create Heading and Introduction sections of Project 02

Project 2:  Working with the Titanic Dataset
Author:  James Pinkston  
Date:  November 1, 2025  
Objective:  P2: Exploratory Data Analysis and Machine Learning Preparation of the Titanic Dataset.

## Create Imports section of Project 02

Imports
```shell
# All imports should be at the top of the notebook

# Import seaborn for statistical data visualization (built on matplotlib)
import seaborn as sns

# Import pandas for data manipulation and analysis
import pandas as pd

# Import scatter matrix
from pandas.plotting import scatter_matrix

# Import matplotlib for creating static visualizations
import matplotlib.pyplot as plt

# Import train_test_split for basic train/test split
from sklearn.model_selection import train_test_split

# Import StratifiedShuffleSplit for stratified train/test split
from sklearn.model_selection import StratifiedShuffleSplit
```

## Create Section 1 of Project 02

Section 1. Import and Inspect the Data

### 1.1 Load the Titanic dataset from the seaborn library for consistency and simplicity:
```shell
# Load Titanic dataset

titanic = sns.load_dataset('titanic')
```

### 1.2 Display basic information about the dataset using the info() method:
```shell
titanic.info()
```
```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
```

### 1.3 Display the first 10 rows:
```shell
print(titanic.head(10))
```
```text
survived  pclass     sex   age  sibsp  parch     fare embarked   class  \
0         0       3    male  22.0      1      0   7.2500        S   Third   
1         1       1  female  38.0      1      0  71.2833        C   First   
2         1       3  female  26.0      0      0   7.9250        S   Third   
3         1       1  female  35.0      1      0  53.1000        S   First   
4         0       3    male  35.0      0      0   8.0500        S   Third   
5         0       3    male   NaN      0      0   8.4583        Q   Third   
6         0       1    male  54.0      0      0  51.8625        S   First   
7         0       3    male   2.0      3      1  21.0750        S   Third   
8         1       3  female  27.0      0      2  11.1333        S   Third   
9         1       2  female  14.0      1      0  30.0708        C  Second   

     who  adult_male deck  embark_town alive  alone  
0    man        True  NaN  Southampton    no  False  
1  woman       False    C    Cherbourg   yes  False  
2  woman       False  NaN  Southampton   yes   True  
3  woman       False    C  Southampton   yes  False  
4    man        True  NaN  Southampton    no   True  
5    man        True  NaN   Queenstown    no   True  
6    man        True    E  Southampton    no   True  
7  child       False  NaN  Southampton    no  False  
8  woman       False  NaN  Southampton   yes  False  
9  child       False  NaN    Cherbourg   yes  False
```

### 1.4 Check for missing values using the isnull() method and then the sum() method:
```shell
titanic.isnull().sum()
```
```text
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
```

### 1.5 Display summary statistics using the desribe() method:
```shell
print(titanic.describe())
```
```text
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
```

### 1.6 Check for correlations using the corr() method and tell it to use only the numeric features:
```shell
print(titanic.corr(numeric_only=True))
```
```text
            survived    pclass       age     sibsp     parch      fare  \
survived    1.000000 -0.338481 -0.077221 -0.035322  0.081629  0.257307   
pclass     -0.338481  1.000000 -0.369226  0.083081  0.018443 -0.549500   
age        -0.077221 -0.369226  1.000000 -0.308247 -0.189119  0.096067   
sibsp      -0.035322  0.083081 -0.308247  1.000000  0.414838  0.159651   
parch       0.081629  0.018443 -0.189119  0.414838  1.000000  0.216225   
fare        0.257307 -0.549500  0.096067  0.159651  0.216225  1.000000   
adult_male -0.557080  0.094035  0.280328 -0.253586 -0.349943 -0.182024   
alone      -0.203367  0.135207  0.198270 -0.584471 -0.583398 -0.271832   

            adult_male     alone  
survived     -0.557080 -0.203367  
pclass        0.094035  0.135207  
age           0.280328  0.198270  
sibsp        -0.253586 -0.584471  
parch        -0.349943 -0.583398  
fare         -0.182024 -0.271832  
adult_male    1.000000  0.404744  
alone         0.404744  1.000000  
```

## Create Reflection 1 for Section 1

```text
Reflection 1

1) How many data instances are there? **891**

2) How many features are there? **15**

3) What are the names? **survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who, adult_male, deck, embark_town, alive, alone**

4) Are there any missing values? **Yes: age - 177, embarked - 2, deck - 688, embark_town - 2**

5) Are there any non-numeric features? **Yes: object - sex, embarked, who, embark_town, alive; category - class, deck; boolean - adult_male, alone**

6) Are the data instances sorted on any of the attributes? **No, the data is not sored on any attribute.**

7) What are the units of age? **Years**

8) What are the minimum, median, and max age? **minimum: 0.42, media: 28.00, max: 80.00**

9) What two different features have the highest correlation? **sibsp - parch: 0.414838 | adult_male - alone: 0.404744**

10) Are there any categorical features that might be useful for prediction? **Yes: sex, who, alone and pclass - these categories could all potentially be used to predict survived**
```

## Create Section 2 of Project 02

Section 2. Data Exploration and Preparation

### 2.1 Explore Data Patterns and Distributions

Create a scatter matrix using only numeric attributes:

```shell
attributes = ['age', 'fare', 'pclass']
scatter_matrix(titanic[attributes], figsize=(10, 10))
```
```text
array([[<Axes: xlabel='age', ylabel='age'>,
        <Axes: xlabel='fare', ylabel='age'>,
        <Axes: xlabel='pclass', ylabel='age'>],
       [<Axes: xlabel='age', ylabel='fare'>,
        <Axes: xlabel='fare', ylabel='fare'>,
        <Axes: xlabel='pclass', ylabel='fare'>],
       [<Axes: xlabel='age', ylabel='pclass'>,
        <Axes: xlabel='fare', ylabel='pclass'>,
        <Axes: xlabel='pclass', ylabel='pclass'>]], dtype=object)
```

Create a scatter plot of age vs fare, colored by gender:
```shell
plt.scatter(titanic['age'], titanic['fare'], c=titanic['sex'].apply(lambda x: 0 if x == 'male' else 1))
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare by Gender')
plt.show()
```

Create a histogram of age:
```shell
sns.histplot(titanic['age'], kde=True)
plt.title('Age Distribution')
plt.show()
```

Create a count plot for class and survival:
```shell
sns.countplot(x='class', hue='survived', data=titanic)
plt.title('Class Distribution by Survival')
plt.show()
```

### Create a Reflection for Section 2.1

```text
Reflect 2.1

1) What patterns or anomalies do you notice? **Most passengers are between the ages of 20-40, most passengers paid a fare of 100 or less, and most Third Class passengers did not survive. Second Class survivability was nearly 50%, and a higher percentage of First Class passengers surived then did not survive.**

2) Do any features stand out as potential predictors? **Higher fares correlate with First Class passengers, who had a higher survival rate. Females and children likely had a higher survival rate since females and children were allowed to board lifeboats first. Since the ship filled with water from the bottom up, lower fare passengers (as shown in the plot) had a higher chance to not survive since their berthing areas were in the lower part of the ship.**

3) Are there any visible class inbalances? **Most passengers paid less than 100, which corresponds to Third Class passengers. We know from historical data not presented here that their berthing areas were near the bottom of the ship, and thus had a much lower survival rate.**
```

### 2.2 Handle Missing Values and Clean Data

Age was missing values. We can impute missing values for age using the media:
```shell
titanic['age'].fillna(titanic['age'].median(), inplace=True)
```

Embark_town was missing values. We can drop missing values for embark_town (or fill with mode):
```shell
titanic['embark_town'].fillna(titanic['embark_town'].mode()[0], inplace=True)
```

### 2.3 Feature Engineering

Create a new feature: Family size
```shell
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
```

Convert categorical data to numeric:
```shell
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
```

Create a binary feature for 'alone':
```shell
titanic['alone'] = titanic['alone'].astype(int)
```

### Create a Reflection for Section 2.3

```text
Reflection 2.3

1) Why might family size be a useful feature for predicting survival? **Women and children were allowed to board the lifeboats first, and IIRC, single men were nearly the last group allowed to board the lifeboats.**

2) Why convert categorical data to numeric? **Linear regression and other types of machine learning models cannot handle string labels and need numeric inputs to calculate sums, averages, probabilities, or splits.**
```

## Create Section 3 of Project 02

### 3.1 Choose Features and Target

- Select two or more input feature (numerical for regression, numerical and/or categorical for classification)
- Select a target variable (as applicable)
  - Classification: Categorical target variable (e.g., gender, species).
  - Justify your selection with reasoning.

For classification, we'll use **survived** as the target variable.

Input features: age, fare, pclass, family_size
Target: survived

### 3.2 Define X and y

- Assign input features to X
- Assign target variable to y (as applicable)

```shell
X = titanic[['age', 'fare', 'pclass', 'sex', 'family_size']]
y = titanic['survived']
```

## Create a Reflection for Section 3

```text
Reflection 3

1) Why are these features selected? **These features are numerical or have been encoded as numeric, which lends themselves to being interpreted by machine learning models. They are also relevant to survival.**

2) Are there any features that are likely to be highly predictive of survival? **Fare, Sex, age, and family_size are likely to be highly predictive as women and children were given priority for boarding lifeboats. Also, as we have seen with the plots, cheaper fares had an extremely low survival rate.**
```

## Create Section 4 of Project 02

Split the data into training and test sets using train_split first and StratifiedShuffleSplit second. Compare.

### 4.1 Basic Train/Test Split

```shell
full_data = pd.concat([X, y], axis=1)
train_set, test_set = train_test_split(full_data, test_size=0.2, random_state=123)
print('Train Size:', len(train_set))
print('Test Size:', len(test_set))
```
```text
Train Size: 712
Test Size: 179
```

### 4.2 Stratified Train/Test Split

```shell
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)

for train_indices, test_indices in splitter.split(X, y):
    train_set = X.iloc[train_indices]
    test_set = X.iloc[test_indices]
print('Train Size:', len(train_set))
print('Test Size:', len(test_set))
```
```text
Train Size: 712
Test Size: 179
```

### 4.3 Compare Results

```shell
print("Original Class Distribution:\n", y.value_counts(normalize=True))
print("Train Set Class Distribution:\n", train_set['pclass'].value_counts(normalize=True))
print("Test Set Class Distribution:\n", test_set['pclass'].value_counts(normalize=True))
```

## Create a Reflection for Section 4

```text
Reflection 4

1) Why might stratification improve model performance? **Train/test sets better reflect the true class balance.**

2) How close are the training and test distributions to the original dataset? **Train and test sets are reasonably close to the original dataset, though minor differences exist.**

3) Which split method produced better class balance? **The stratified split is better because it preserves the original distribution in both train and test sets. The basic split is random, so the train/test sets could have slightly distored distributions.**
```