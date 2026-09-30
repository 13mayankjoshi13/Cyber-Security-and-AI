import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# -------------------------------
# 1. Create Dataset
# -------------------------------

data = {
    "Age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 100],
    "Salary": [25000, 30000, 28000, 35000, 40000,
               42000, 45000, 48000, 50000, 200000],
    "Experience": [1, 2, 1, 3, 4, 5, 5, 6, 7, 10],
    "City": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai",
             "Delhi", "Pune", "Delhi", "Mumbai", "Delhi"],
    "Purchased": [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# -------------------------------
# 2. Remove Duplicate Rows
# -------------------------------

df = df.drop_duplicates()

# -------------------------------
# 3. Detect Outliers using IQR
# -------------------------------

numeric_columns = ["Age", "Salary", "Experience"]

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df[column] = df[column].clip(lower_limit, upper_limit)

print("\nDataset after Outlier Handling:")
print(df)

# -------------------------------
# 4. Separate Features and Target
# -------------------------------

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# -------------------------------
# 5. Define Columns
# -------------------------------

numeric_features = ["Age", "Salary", "Experience"]
categorical_features = ["City"]

# -------------------------------
# 6. Numerical Pipeline
# -------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# -------------------------------
# 7. Categorical Pipeline
# -------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# -------------------------------
# 8. Combine Preprocessing
# -------------------------------

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# -------------------------------
# 9. Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 10. Apply Preprocessing
# -------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Data:")
print(X_train_processed.toarray())

print("\nProcessed Testing Data:")
print(X_test_processed.toarray())

print("\nTraining Data Shape:", X_train_processed.shape)
print("Testing Data Shape:", X_test_processed.shape)
