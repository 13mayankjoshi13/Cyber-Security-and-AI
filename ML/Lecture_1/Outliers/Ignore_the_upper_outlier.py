"""Now suppose we have:

Normal data:
(1,20)
(2,30)
(3,40)
(4,50)
(5,60)

Outlier:
(6,120)

But we don't include the outlier when training."""


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Normal data
X = [[1], [2], [3], [4], [5]]
Y = [20, 30, 40, 50, 60]

# Outlier
outlier_X = [[6]]
outlier_Y = [120]

# Train only on normal data
model = LinearRegression()
model.fit(X, Y)

# Slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Regression line
Y_pred = model.predict(X)

# Plot normal data
plt.scatter(X, Y, label="Normal Data")

# Plot ignored outlier
plt.scatter(outlier_X, outlier_Y, label="Ignored Outlier")

# Plot regression line
plt.plot(X, Y_pred, label="Regression Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Upper Outlier Ignored")
plt.legend()

plt.show()
