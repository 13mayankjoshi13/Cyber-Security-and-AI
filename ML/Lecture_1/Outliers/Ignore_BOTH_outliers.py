T"""This is the most useful comparison.

Suppose we have:

Normal:
(1,20)
(2,30)
(3,40)
(4,50)
(5,60)

Upper outlier:
(6,120)

Lower outlier:
(7,-40)

We can display both outliers but train the model only on normal data."""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Normal data
X = [[1], [2], [3], [4], [5]]
Y = [20, 30, 40, 50, 60]

# Two outliers
upper_X = [[6]]
upper_Y = [120]

lower_X = [[7]]
lower_Y = [-40]

# Train only on normal data
model = LinearRegression()
model.fit(X, Y)

# Slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)

# Regression line
Y_pred = model.predict(X)

# Normal points
plt.scatter(X, Y, label="Normal Data")

# Outliers
plt.scatter(upper_X, upper_Y, label="Upper Outlier")
plt.scatter(lower_X, lower_Y, label="Lower Outlier")

# Regression line
plt.plot(X, Y_pred, label="Regression Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Both Outliers Ignored")
plt.legend()

plt.show()
