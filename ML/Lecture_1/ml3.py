import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Training data
X = [[1], [2], [3], [4], [5]]
Y = [20, 30, 40, 50, 60]

# Create model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# Get slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)

# New X value
new_x = [[6]]

# Predict Y
prediction = model.predict(new_x)

print("New X:", new_x[0][0])
print("Predicted Y:", prediction[0])

# -------------------------
# Plotting
# -------------------------

# Predict Y values for the training X values
Y_pred = model.predict(X)

# Plot original data points
plt.scatter(X, Y)

# Plot regression line
plt.plot(X, Y_pred)

# Plot the new predicted point
plt.scatter(new_x, prediction)

# Labels
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")

# Show graph
plt.show()
