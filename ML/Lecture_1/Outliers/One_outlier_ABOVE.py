import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Normal data
X = [[1], [2], [3], [4], [5]]
Y = [20, 30, 40, 50, 60]

# Add an outlier ABOVE
X_outlier = X + [[6]]
Y_outlier = Y + [120]

# Create model
model = LinearRegression()

# Train with outlier
model.fit(X_outlier, Y_outlier)

# Get slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)

# Predictions
Y_pred = model.predict(X_outlier)

# Plot normal data
plt.scatter(X, Y, label="Normal Data")

# Plot outlier
plt.scatter([[6]], [120], label="Outlier")

# Plot regression line
plt.plot(X_outlier, Y_pred, label="Regression Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Outlier Above")
plt.legend()

plt.show()
