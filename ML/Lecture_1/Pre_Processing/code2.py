import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Dataset
data = {
    "Age": [20, 21, None, 23, 24, 22, None, 25],
    "Salary": [25000, 30000, 28000, None, 40000, 35000, 32000, 45000],
    "City": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai", "Pune", None, "Delhi"],
    "Experience": [1, 2, 1, 3, 4, None, 2, 5],
    "Purchased": [0, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Features and target
X = df.drop("Purchased", axis=1)
y = df["Purchased"]

# Numerical and categorical columns
num_cols = ["Age", "Salary", "Experience"]
cat_cols = ["City"]

# Numerical preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("numerical", num_pipeline, num_cols),
    ("categorical", cat_pipeline, cat_cols)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Data:")
print(X_train_processed.toarray())

print("\nProcessed Testing Data:")
print(X_test_processed.toarray())
