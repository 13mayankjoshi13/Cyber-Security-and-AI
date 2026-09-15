from sklearn.linear_model import LinearRegression

# X and Y values
X = [[1], [2], [3], [4], [5]]
Y = [20, 30, 40, 50, 60]

# Create and train model
model = LinearRegression()
model.fit(X, Y)

# Get slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

print("Slope (m):", slope)
print("Intercept (c):", intercept)

# New X value
new_x = [[6]]

# Predict Y
new_y = model.predict(new_x)

print("New X:", new_x[0][0])
print("Predicted Y:", new_y[0])

#Output
Slope (m): 10.0
Intercept (c): 10.0
New X: 6
Predicted Y: 70.0
