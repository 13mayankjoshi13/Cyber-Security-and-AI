import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    "Age": [20, 21, 22, 23, 24],
    "Salary": [20000, 25000, 30000, 35000, 40000],
    "Study_Hours": [2, 4, 5, 6, 8],
    "Passed": [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Separate features and target
X = df[["Age", "Salary", "Study_Hours"]]
y = df["Passed"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nPreprocessed Training Data:")
print(X_train)

print("\nPreprocessed Testing Data:")
print(X_test)
