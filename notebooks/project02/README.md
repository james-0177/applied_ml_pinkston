# Project 2:  Working with the Titantic Dataset

Project 2 uses the Titantic dataset to perform exploratory data analysis and machine learning preparation. I'll import, inspect, explore, and clean the data; and then I'll split the data for training and testing.

## Activate Project Virtual Environment

Since this project (and future projects for this course) all use the same repository, they will all use the same local virtual environment created for Project 1. All that needs to be done for each project is to activate.

```shell
.\.venv\Scripts\activate
```

## Create new project directory and files with VS Code

Inside the notebooks directory, create a project02 subdirectory with the following files inside:
ml02_pinkston.ipynb
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

## Create Heading and Introduction sections of Project 2

Project 2:  Working with the Titanic Dataset
Author:  James Pinkston  
Date:  November 1, 2025  
Objective:  P2: Exploratory Data Analysis and Machine Learning Preparation of the Titanic Dataset.

## Create Imports section of Project 2

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

## Create Section 1 of Project 2

Section 1. Import and Inspect the Data

1.1 Load the Titanic dataset from the seaborn library for consistency and simplicity:
```shell
# Load Titanic dataset

titanic = sns.load_dataset('titanic')
```

1.2 Display basic information about the dataset using the info() method:
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

1.3 Display the first 10 rows:
```shell
print(titanic.head(10))
```
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

1.4 Check for missing values using the isnull() method and then the sum() method:
```shell
titanic.isnull().sum()
```
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

1.5 Display summary statistics using the desribe() method:
```shell
print(titanic.describe())
```
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

1.6 Check for correlations using the corr() method and tell it to use only the numeric features:
```shell
print(titanic.corr(numeric_only=True))
```
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

## Create Reflection 1 for Section 1

