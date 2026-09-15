from sklearn.linear_model import LinearRegression

# Training data
X = [[1], [2], [3], [4], [5]]
Y = [20, 30, 40, 50, 60]

# Create model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# New X value
new_x = [[6]]

# Predict Y
prediction = model.predict(new_x)

print("Predicted Y:", prediction[0])

#OUTPUT
Predicted Y: 70.0
