from sklearn.tree import DecisionTreeClassifier

# [Study Hours, Attendance]
X = [
    [2, 60],
    [3, 65],
    [4, 70],
    [6, 80],
    [7, 90]
]

# 0 = Fail, 1 = Pass
y = [0, 0, 0, 1, 1]

model = DecisionTreeClassifier()
model.fit(X, y)

# New student's data
result = model.predict([[5, 75]])

print("Pass" if result[0] == 1 else "Fail")
