# Heart Attack Prediction Dataset Analysis

## Step 1: Data Loading, Inspection, and Cleaning

### Dataset Overview
The dataset contains information about various health and lifestyle factors that may contribute to the risk of heart attacks. The dataset includes 8,763 entries and 26 columns, with data points such as age, cholesterol levels, blood pressure, diabetes status, and more.

### Dataset Columns
The dataset includes the following columns:
1. `Patient ID` - Unique identifier for each patient
2. `Age` - Age of the patient
3. `Sex` - Gender of the patient
4. `Cholesterol` - Cholesterol level
5. `Blood Pressure` - Blood pressure reading
6. `Heart Rate` - Heart rate in beats per minute
7. `Diabetes` - Diabetes status (binary)
8. `Family History` - Family history of heart disease (binary)
9. `Smoking` - Smoking status (binary)
10. `Obesity` - Obesity status (binary)
11. `Alcohol Consumption` - Frequency of alcohol consumption
12. `Exercise Hours Per Week` - Hours of exercise per week
13. `Diet` - Diet quality
14. `Previous Heart Problems` - History of heart problems (binary)
15. `Medication Use` - Medication usage (binary)
16. `Stress Level` - Stress level
17. `Sedentary Hours Per Day` - Hours spent sedentary per day
18. `Income` - Annual income
19. `BMI` - Body Mass Index
20. `Triglycerides` - Triglyceride levels
21. `Physical Activity Days Per Week` - Days of physical activity per week
22. `Sleep Hours Per Day` - Hours of sleep per day
23. `Country` - Country of residence
24. `Continent` - Continent of residence
25. `Hemisphere` - Hemisphere of residence
26. `Heart Attack Risk` - Risk of heart attack (binary)

### Loading the Dataset
The dataset is loaded into a pandas DataFrame using the `pd.read_csv` function. The file path to the dataset is provided as `~/Downloads/heart_attack_prediction_dataset.csv`.

```python
import pandas as pd

# Load the dataset
file_path = "~/Downloads/heart_attack_prediction_dataset.csv"
df = pd.read_csv(file_path)
```

### Initial Inspection
Basic information about the dataset is displayed, including its shape, data types, and the first few rows of the dataset.

```python
# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nDataset Head:")
print(df.head())
```

### Missing Values
The dataset is checked for missing values. In this case, no missing values are found.

```python
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
```

### Duplicate Rows
The dataset is checked for duplicate rows. No duplicate rows are found in the dataset.

```python
# Check for duplicates
print("\nDuplicate Rows:")
print(df.duplicated().sum())
```

### Handling Missing Values
For demonstration purposes, any rows with missing values are dropped. However, since no missing values were found, this step does not alter the dataset in this case.

```python
# Handle missing values
# For demonstration purposes, let's drop rows with missing values
df.dropna(inplace=True)
```

### Summary
- The dataset is successfully loaded and inspected.
- There are 8,763 entries and 26 columns.
- No missing values or duplicate rows are present in the dataset.
- The dataset is ready for further analysis and processing.

---

## Step 2: Data Preprocessing and Feature Engineering

### Introduction
In this step, we preprocess the dataset and perform feature engineering to prepare the data for modeling. This includes handling categorical variables, normalizing numerical features, and creating new features if necessary.

### Handling Categorical Variables
Categorical variables such as `Sex`, `Country`, `Continent`, `Hemisphere`, and `Diet` need to be converted into numerical format for model compatibility. This can be done using one-hot encoding.

```python
# One-hot encode categorical variables
categorical_columns = ['Sex', 'Country', 'Continent', 'Hemisphere', 'Diet']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
```

### Normalizing Numerical Features
Numerical features are normalized to ensure they are on a similar scale, which helps in improving the performance of machine learning models.

```python
from sklearn.preprocessing import StandardScaler

# Define numerical columns to be normalized
numerical_columns = ['Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week', 
                     'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides', 
                     'Physical Activity Days Per Week', 'Sleep Hours Per Day']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
```

### Splitting the Dataset
The dataset is split into training and testing sets to evaluate the performance of the model. The target variable is `Heart Attack Risk`.

```python
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop(columns=['Patient ID', 'Heart Attack Risk'])
y = df['Heart Attack Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Summary
- Categorical variables are converted to numerical format using one-hot encoding.
- Numerical features are normalized for improved model performance.
- The dataset is split into training and testing sets for model evaluation.

---

## Step 3: Analyzing Categorical Variables

### Introduction
In this step, we analyze the value counts for each categorical variable to understand the distribution of the data.

### Value Counts for Categorical Variables
We print the value counts for the categorical columns `Sex`, `Diet`, `Country`, `Continent`, and `Hemisphere`.

```python
categorical_columns = ['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere']

for column in categorical_columns:
    print(f'Value counts for {column}:')
    print(df[column].value_counts())
    print()
```

#### Output
```
Value counts for Sex:
Male      6111
Female    2652
Name: Sex, dtype: int64

Value counts for Diet:
Healthy      2960
Average      2912
Unhealthy    2891
Name: Diet, dtype: int64

Value counts for Country:
Germany           477
Argentina         471
Brazil            462
United Kingdom    457
Australia         449
Nigeria           448
France            446
Canada            440
China             436
New Zealand       435
Japan             433
Italy             431
Spain             430
Colombia          429
Thailand          428
South Africa      425
Vietnam           425
United States     420
India             412
South Korea       409
Name: Country, dtype: int64

Value counts for Continent:
Asia             2543
Europe           2241
South America    1362
Australia         884
Africa            873
North America     860
Name: Continent, dtype: int64

Value counts for Hemisphere:
Northern Hemisphere    5660
Southern Hemisphere    3103
Name: Hemisphere, dtype: int64
```

### Summary
- The value counts for each categorical column are displayed to understand their distributions.
- This analysis helps in understanding the composition of the dataset based on various categorical features.

---

## Step 4: Data Consistency Checks

### Introduction
In this step, we perform data consistency checks to identify any inconsistencies in the dataset.

### Data Consistency Checks
We check for inconsistencies in the `Age` and `Heart Rate` columns to ensure data quality.

```python
# check: Age should be positive
if (df['Age'] < 0).any():
    print("Inconsistent Age values found.")

#  check: Heart Rate should be within a reasonable range
if (df['Heart Rate'] < 30).any() or (df['Heart Rate'] > 220).any():
    print("Inconsistent Heart Rate values found.")
```

### Summary
- The dataset is checked for inconsistencies in `Age` and `Heart Rate`.
- No inconsistencies are found in the `Age` and `Heart Rate` columns.

#### Output
```
        Patient ID          Age   Sex  Cholesterol Blood Pressure   Heart Rate  \
count        8763  8763.000000  8763  8763.000000           8763  8763.000000   
unique       8763          NaN     2          NaN           3915          NaN   
top       BMW7812          NaN  Male          NaN         146/94          NaN   
freq            1          NaN  6111          NaN              8          NaN   
mean          NaN    53.707977   NaN   259.877211            NaN    75.021682   
std           NaN    21.249509   NaN    80.863276            NaN    20.550948   
min           NaN    18.000000   NaN   120.000000            NaN    40.000000   
25%           NaN    35.000000   NaN   192.000000            NaN    57.000000   
50%           NaN    54.000000   NaN   259.000000            NaN    75.000000   
75%           NaN    72.000000   NaN   330.000000            NaN    93.000000   
max           NaN    90.000000   NaN   400.000000            NaN   110.000000   

           Diabetes  Family History      Smoking      Obesity  ...  \
count   8763.000000     8763.000000  8763.000000  8763.000000  ...   
unique          NaN             NaN          NaN          NaN  ...   
top             NaN             NaN          NaN          NaN  ...   
freq            NaN             NaN          NaN          NaN  ...   
mean       0.652288        0.492982     0.896839     0.501426  ...   
std        0.476271        0.499979     0.304186     0.500026  ...   
min        0.000000        0.000000     0.000000     0.000000  ...   
25%        0.000000        0.000000     1.000000     0.000000  ...   
50%        1.000000        0.000000     1.000000     1.000000  ...   
75%        1.000000        1.000000     1.000000     1.000000  ...   
max        1.000000        1.000000     1.000000     1.000000  ...   

        Sedentary Hours Per Day         Income          BMI  Triglycerides  \
count               8763.000000    8763.000000  8763.000000    8763.000000   
unique                      NaN            NaN          NaN            NaN   
top                         NaN            NaN          NaN            NaN   
freq                        NaN            NaN          NaN            NaN   
mean                   5.993690  158263.181901    28.891446     417.677051   
std                    3.466359   80575.190806     6.319181     223.748137   
min                    0.001263   20062.000000    18.002337      30.000000   
25%                    2.998794   88310.000000    23.422985     225.500000   
50%                    5.933622  157866.000000    28.768999     417.000000   
75%                    9.019124  227749.000000    34.324594     612.000000   
max                   11.999313  299954.000000    39.997211     800.000000   

        Physical Activity Days Per Week  Sleep Hours Per Day  Country  \
count                       8763.000000          8763.000000     8763   
unique                              NaN                  NaN       20   
top                                 NaN                  NaN  Germany   
freq                                NaN                  NaN      477   
mean                           3.489672             7.023508      NaN   
std                            2.282687             1.988473      NaN   
min                            0.000000             4.000000      NaN   
25%                            2.000000             5.000000      NaN   
50%                            3.000000             7.000000      NaN   
75%                            5.000000             9.000000      NaN   
max                            7.000000            10.000000      NaN   

        Continent           Hemisphere  Heart Attack Risk  
count        8763                 8763        8763.000000  
unique          6                    2                NaN  
top          Asia  Northern Hemisphere                NaN  
freq         2543                 5660                NaN  
mean          NaN                  NaN           0.358211  
std           NaN                  NaN           0.479502  
min           NaN                  NaN           0.000000  
25%           NaN                  NaN           0.000000  
50%           NaN                  NaN           0.000000  
75%           NaN                  NaN           1.000000  
max           NaN                  NaN           1.000000  

[11 rows x 26 columns]
```

---

## Step 5: Outlier Detection

### Introduction
In this step, we detect outliers in the numerical columns using the Z-score method.

### Outlier Detection Code
We use the Z-score method to detect outliers in the numerical columns of the dataset.

```python
import pandas as pd
import numpy as np
from scipy import stats

# List of numerical columns
numerical_columns = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 
                     'Exercise Hours Per Week', 'Stress Level', 'Sedentary Hours Per Day', 
                     'Income', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week', 
                     'Sleep Hours Per Day']

# Ensure numerical columns are numeric and handle non-numeric values
for column in numerical_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Function to detect outliers using Z-score
def detect_zscore_outliers(df, column):
    # Re-check column is numeric after coercion
    if df[column].dtype in ['float64', 'int64']:
        df['z_score'] = stats.zscore(df[column])
        outliers = df[df['z_score'].abs() > 3]  # Typically, z-scores > 3 or < -3 are considered outliers
        return outliers
    else:
        print(f"Column {column} is not numeric after coercion")
        return pd.DataFrame()

# Detect and print outliers for each numerical column based on Z-scores
for column in numerical_columns:
    outliers = detect_zscore_outliers(df, column)
    print(f'Outliers in {column} based on Z-scores:')
    if not outliers.empty:
        print(outliers[['z_score', column]])
    else:
        print("No outliers found.")

# Remove the z_score column after analysis
df.drop(columns=['z_score'], inplace=True, errors='ignore')
```

### Summary
- Outliers in numerical columns are detected using the Z-score method.
- No outliers are found in any of the numerical columns.

#### Output
```
Outliers in Age based on Z-scores:
No outliers found.
Outliers in Cholesterol based on Z-scores:
No outliers found.
Outliers in Blood Pressure based on Z-scores:
No outliers found.
Outliers in Heart Rate based on Z-scores:
No outliers found.
Outliers in Exercise Hours Per Week based on Z-scores:
No outliers found.
Outliers in Stress Level based on Z-scores:
No outliers found.
Outliers in Sedentary Hours Per Day based on Z-scores:
No outliers found.
Outliers in Income based on Z-scores:
No outliers found.
Outliers in BMI based on Z-scores:
No

----
## Step 6: Exploratory Data Analysis (EDA) and Correlation Analysis

### Introduction
In this step, we perform Exploratory Data Analysis (EDA) and correlation analysis to understand the relationships between variables in the dataset, especially in relation to heart attack risk.

### EDA and Correlation Analysis

#### Univariate Analysis
We start with univariate analysis to understand the distribution of individual variables.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "~/Downloads/heart_attack_prediction_dataset.csv"
df = pd.read_csv(file_path)

# Univariate Analysis

# Numeric variables
numeric_columns = ['Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week', 'Income', 'BMI']

# Categorical variables
categorical_columns = ['Sex', 'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Diet', 'Previous Heart Problems', 'Medication Use', 'Stress Level', 'Country', 'Continent', 'Hemisphere', 'Heart Attack Risk']

# Visualize numeric variables
for column in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# Visualize categorical variables
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df[column])
    plt.title(f"Countplot of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
```

#### Bivariate Analysis
Next, we perform bivariate analysis to explore relationships between variables and the target variable (heart attack risk).

```python
# Scatter plots for numeric variables against target variable
for column in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=column, y='Heart Attack Risk', data=df)
    plt.title(f"Scatter plot of {column} vs Heart Attack Risk")
    plt.xlabel(column)
    plt.ylabel("Heart Attack Risk")
    plt.show()

# Box plots for categorical variables against target variable
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=column, y='Heart Attack Risk', data=df)
    plt.title(f"Box plot of {column} vs Heart Attack Risk")
    plt.xlabel(column)
    plt.ylabel("Heart Attack Risk")
    plt.xticks(rotation=45)
    plt.show()
```

#### Correlation Matrix
We create a correlation matrix to see the relationships between all variables and the heart attack risk.

```python
# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix including all variables and heart risk prediction
correlation_matrix = df.corr()

# Filter the correlation of 'heart risk prediction' with all other variables
heart_risk_correlation = correlation_matrix['Heart Attack Risk']

# Plot the correlation of 'heart risk prediction' with all other variables
plt.figure(figsize=(10, 8))
sns.heatmap(heart_risk_correlation.to_frame(), annot=True, cmap='coolwarm', cbar=False)
plt.title('Correlation of Heart Risk Prediction with Other Variables')
plt.show()
```

#### Top Correlated Variables
We identify the top 5 variables that are most strongly correlated with heart attack risk.

```python
import numpy as np

# Calculate correlation coefficients between 'heart risk prediction' and other variables
correlation_with_heart_risk = df.corr()['Heart Attack Risk']

# Sort the correlation coefficients by their absolute values
sorted_correlation = correlation_with_heart_risk.abs().sort_values(ascending=False)

# Extract top 5 variables (excluding 'heart risk prediction' itself)
top_variables = sorted_correlation[1:6]

# Create a bar plot to visualize the correlation coefficients
plt.figure(figsize=(10, 6))
top_variables.plot(kind='bar', color='skyblue')
plt.title('Top 5 Correlated Variables with Heart Risk Prediction')
plt.xlabel('Variables')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

#### Scatter Plots of Top Variables
We create scatter plots of the top correlated variables against heart attack risk for a more detailed view.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Selecting numerical columns for analysis
numerical_columns = ['Age', 'Cholesterol', 'Heart Rate', 'BMI', 'Triglycerides']

# Creating scatter plots for each numerical column against Heart Attack Risk
plt.figure(figsize=(18, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(2, 3, i)
    sns.regplot(x=column, y='Heart Attack Risk', data=df, scatter_kws={'alpha':0.3}, line_kws={'color': 'red'})
    plt.title(f'{column} vs Heart Attack Risk')
    plt.xlabel(column)
    plt.ylabel('Heart Attack Risk')
plt.tight_layout()
plt.show()
```

### Summary
- Performed univariate analysis to understand the distribution of individual variables.
- Conducted bivariate analysis to explore relationships between variables and heart attack risk.
- Created a correlation matrix to see the relationships between all variables and heart attack risk.
- Identified the top 5 variables most strongly correlated with heart attack risk.
- Created scatter plots of the top correlated variables against heart attack risk.

---

## Step 7: Feature Selection Using Recursive Feature Elimination (RFE)

### Introduction
In this step, we clean the dataset by handling non-numeric data, imputing missing values, and performing feature selection using Recursive Feature Elimination (RFE).

### Code for Feature Selection

```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('~/Downloads/heart_attack_prediction_dataset.csv')

# Print the first few rows to inspect the data
print(df.head())

# Identify and clean non-numeric data in numeric columns
for col in df.columns:
    if df[col].dtype == object:
        # Print unique values for columns with object type to inspect them
        print(f"Column '{col}' unique values: {df[col].unique()}")
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except ValueError:
            pass

# Fill missing values after conversion
df.fillna(df.mean(), inplace=True)

# Check for columns with all missing values
missing_all = df.columns[df.isna().all()].tolist()
print(f"Columns with all missing values: {missing_all}")

# Drop columns with all missing values
df.drop(columns=missing_all, inplace=True)

# Separate target and features
X = df.drop('Heart Attack Risk', axis=1)
y = df['Heart Attack Risk']

# Handling non-numeric columns
# Separate features into numeric and categorical
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=[object]).columns.tolist()

# Remove 'Heart Attack Risk' from features list if still present
if 'Heart Attack Risk' in numeric_features:
    numeric_features.remove('Heart Attack Risk')
if 'Heart Attack Risk' in categorical_features:
    categorical_features.remove('Heart Attack Risk')

# Define the column transformer with an imputer for missing values
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Define the model with increased max_iter and solver
model = LogisticRegression(max_iter=1000, solver='saga')

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# Fit the model using RFE for feature selection
rfe = RFE(model, n_features_to_select=1)

# Fit the pipeline
pipeline.fit(X, y)

# Fit RFE
rfe.fit(X, y)

# Get the ranking of features
ranking = rfe.ranking_

# Create a DataFrame for the feature ranking
feature_ranking = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': ranking
})

# Sort the DataFrame by ranking
feature_ranking.sort_values(by='Ranking', inplace=True)

# Display the ranking of features
print(feature_ranking)
```

### Result of Feature Selection

```plaintext
  Patient ID  Age     Sex  Cholesterol Blood Pressure  Heart Rate  Diabetes  \
0    BMW7812   67    Male          208         158/88          72         0   
1    CZE1114   21    Male          389         165/93          98         1   
2    BNI9906   21  Female          324         174/99          72         1   
3    JLN3497   84    Male          383        163/100          73         1   
4    GFO8847   66    Male          318          91/88          93         1   

   Family History  Smoking  Obesity  ...  Sedentary Hours Per Day  Income  \
0               0        1        0  ...                 6.615001  261404   
1               1        1        1  ...                 4.963459  285768   
2               0        0        0  ...                 9.463426  235282   
3               1        1        0  ...                 7.648981  125640   
4               1        1        1  ...                 1.514821  160555   

         BMI  Triglycerides  Physical Activity Days Per Week  \
0  31.251233            286                                0   
1  27.194973            235                                1   
2  28.176571            587                                4   
3  36.464704            378                                3   
4  21.809144            231                                1   

   Sleep Hours Per Day    Country      Continent           Hemisphere  \
0                    6  Argentina  South America  Southern Hemisphere   
1                    7     Canada  North America  Northern Hemisphere   
2                    4     France         Europe  Northern Hemisphere   
3                    4     Canada  North America  Northern Hemisphere

----

## Step 8: Model Training and Evaluation

### Introduction
In this step, we will define and train three different machine learning models—Logistic Regression, Random Forest, and Support Vector Machine (SVM). We will then evaluate these models using metrics such as accuracy, precision, recall, and F1 score.

### Code for Model Training and Evaluation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Evaluate models
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

# Print results
results_df = pd.DataFrame(results).T
print(results_df)
```

### Result of Model Evaluation

```plaintext
                      Accuracy  Precision    Recall  F1 Score
Logistic Regression  0.641757   0.411852  0.641757  0.501721
Random Forest        0.636052   0.555936  0.636052  0.518111
SVM                  0.641187   0.531239  0.641187  0.502487
```

### Analysis of Results
- **Logistic Regression**: Accuracy of 64.18%, precision of 41.18%, recall of 64.18%, and F1 score of 50.17%.
- **Random Forest**: Accuracy of 63.61%, precision of 55.59%, recall of 63.61%, and F1 score of 51.81%.
- **SVM**: Accuracy of 64.12%, precision of 53.12%, recall of 64.12%, and F1 score of 50.25%.

### Conclusion
The Logistic Regression and SVM models show similar performance in terms of accuracy, recall, and F1 score, while the Random Forest model has a slightly higher precision but a similar overall performance. Further tuning of model hyperparameters or using different feature engineering techniques might improve the model performance.

----

## Step 9: Model Evaluation with Top Features

### Introduction
In this step, we evaluate the performance of different machine learning models trained on various combinations of the top 5 features identified in the dataset. We explore how the inclusion of different features impacts the accuracy of each model.

### Code for Model Evaluation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Define models to try
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Select top 5 features based on ranking
top_features = feature_ranking['Feature'][:5].tolist()

# Train models on different combinations of top 5 features
for i in range(1, len(top_features) + 1):
    selected_features = top_features[:i]
    X_selected = X[selected_features]
    
    # Print selected features
    print(f"Selected Features: {selected_features}")
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_selected, y)
        # Print model performance metrics
        print(f"Model: {name} - Trained on: {i} features - Accuracy: {model.score(X_selected, y)}")
```

### Result of Model Evaluation

```plaintext
Selected Features: ['Diabetes']
Model: Logistic Regression - Trained on: 1 features - Accuracy: 0.6417893415496976
Model: Random Forest - Trained on: 1 features - Accuracy: 0.6417893415496976
Model: Support Vector Machine - Trained on: 1 features - Accuracy: 0.6417893415496976
Model: K-Nearest Neighbors - Trained on: 1 features - Accuracy: 0.6417893415496976
Model: Decision Tree - Trained on: 1 features - Accuracy: 0.6417893415496976

Selected Features: ['Diabetes', 'Alcohol Consumption']
Model: Logistic Regression - Trained on: 2 features - Accuracy: 0.6417893415496976
Model: Random Forest - Trained on: 2 features - Accuracy: 0.6417893415496976
Model: Support Vector Machine - Trained on: 2 features - Accuracy: 0.6417893415496976
Model: K-Nearest Neighbors - Trained on: 2 features - Accuracy: 0.6417893415496976
Model: Decision Tree - Trained on: 2 features - Accuracy: 0.6417893415496976

Selected Features: ['Diabetes', 'Alcohol Consumption', 'Obesity']
Model: Logistic Regression - Trained on: 3 features - Accuracy: 0.6417893415496976
Model: Random Forest - Trained on: 3 features - Accuracy: 0.6417893415496976
Model: Support Vector Machine - Trained on: 3 features - Accuracy: 0.6417893415496976
Model: K-Nearest Neighbors - Trained on: 3 features - Accuracy: 0.6417893415496976
Model: Decision Tree - Trained on: 3 features - Accuracy: 0.6417893415496976

Selected Features: ['Diabetes', 'Alcohol Consumption', 'Obesity', 'Smoking']
Model: Logistic Regression - Trained on: 4 features - Accuracy: 0.6417893415496976
Model: Random Forest - Trained on: 4 features - Accuracy: 0.6417893415496976
Model: Support Vector Machine - Trained on: 4 features - Accuracy: 0.6417893415496976
Model: K-Nearest Neighbors - Trained on: 4 features - Accuracy: 0.6366541138879379
Model: Decision Tree - Trained on: 4 features - Accuracy: 0.6417893415496976

Selected Features: ['Diabetes', 'Alcohol Consumption', 'Obesity', 'Smoking', 'Sleep Hours Per Day']
Model: Logistic Regression - Trained on: 5 features - Accuracy: 0.6417893415496976
Model: Random Forest - Trained on: 5 features - Accuracy: 0.6428163870820496
Model: Support Vector Machine - Trained on: 5 features - Accuracy: 0.6417893415496976
Model: K-Nearest Neighbors - Trained on: 5 features - Accuracy: 0.5838183270569439
Model: Decision Tree - Trained on: 5 features - Accuracy: 0.6429305032523108
```

### Analysis of Results
- The models exhibit similar accuracies across different combinations of the top 5 features.
- Random Forest and Decision Tree models show slightly higher accuracies compared to other models.
- Including more features does not significantly improve model accuracy, and in some cases, it decreases due to overfitting.

### Conclusion
The evaluation suggests that the selected features have a limited impact on model accuracy. Further exploration and feature engineering may be necessary to improve model performance.
--
```markdown
## Step 10: Run the Gradio Interface

### Introduction
After training the model and saving the pipeline, you can deploy a Gradio interface to interactively predict heart attack risk based on user input.

### Code to Run the Gradio Interface

```python
import pandas as pd
import joblib
import gradio as gr

# Load the trained model pipeline
pipeline = joblib.load('trained_model_pipeline.joblib')

# Define the prediction function
def predict_heart_attack(
        patient_id, age, sex, cholesterol, blood_pressure, heart_rate, diabetes, 
        family_history, smoking, obesity, alcohol_consumption, exercise_hours, diet, 
        previous_heart_problems, medication_use, stress_level, sedentary_hours, income, 
        bmi, triglycerides, physical_activity, sleep_hours, country, continent, hemisphere
    ):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'Patient ID': [patient_id],
        'Age': [age],
        'Sex': [sex],
        'Cholesterol': [cholesterol],
        'Blood Pressure': [blood_pressure],
        'Heart Rate': [heart_rate],
        'Diabetes': [diabetes],
        'Family History': [family_history],
        'Smoking': [smoking],
        'Obesity': [obesity],
        'Alcohol Consumption': [alcohol_consumption],
        'Exercise Hours Per Week': [exercise_hours],
        'Diet': [diet],
        'Previous Heart Problems': [previous_heart_problems],
        'Medication Use': [medication_use],
        'Stress Level': [stress_level],
        'Sedentary Hours Per Day': [sedentary_hours],
        'Income': [income],
        'BMI': [bmi],
        'Triglycerides': [triglycerides],
        'Physical Activity Days Per Week': [physical_activity],
        'Sleep Hours Per Day': [sleep_hours],
        'Country': [country],
        'Continent': [continent],
        'Hemisphere': [hemisphere]
    })

    # Make prediction
    prediction = pipeline.predict(input_data)
    
    # Return the prediction
    return "High Risk" if prediction == 1 else "Low Risk"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_heart_attack,
    inputs=[
        gr.Textbox(label="Patient ID"),
        gr.Number(label="Age"),
        gr.Radio(['Male', 'Female'], label="Sex"),
        gr.Number(label="Cholesterol"),
        gr.Textbox(label="Blood Pressure"),
        gr.Number(label="Heart Rate"),
        gr.Checkbox(label="Diabetes"),
        gr.Checkbox(label="Family History"),
        gr.Checkbox(label="Smoking"),
        gr.Checkbox(label="Obesity"),
        gr.Checkbox(label="Alcohol Consumption"),
        gr.Number(label="Exercise Hours Per Week"),
        gr.Dropdown(['Healthy', 'Average', 'Unhealthy'], label="Diet"),
        gr.Checkbox(label="Previous Heart Problems"),
        gr.Checkbox(label="Medication Use"),
        gr.Number(label="Stress Level"),
        gr.Number(label="Sedentary Hours Per Day"),
        gr.Number(label="Income"),
        gr.Number(label="BMI"),
        gr.Number(label="Triglycerides"),
        gr.Number(label="Physical Activity Days Per Week"),
        gr.Number(label="Sleep Hours Per Day"),
        gr.Textbox(label="Country"),
        gr.Textbox(label="Continent"),
        gr.Textbox(label="Hemisphere")
    ],
    outputs=gr.Textbox(label="Heart Attack Risk Prediction"),
    title="Heart Attack Risk Predictor",
    description="Enter the details to predict the risk of heart attack."
)

# Launch the Gradio interface
interface.launch()
```

### Contributing

My main issue was to make the Gradio interface give accurate results related to my trained model, as it always gave me an error.

### License

This project took the Dataset from Kaggle.

### Acknowledgments

This project uses the following libraries:
- pandas
- scikit-learn
- joblib
- gradio
```
