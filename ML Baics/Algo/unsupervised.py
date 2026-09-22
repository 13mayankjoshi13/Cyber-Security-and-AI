from sklearn.cluster import KMeans

# [Study Hours, Attendance]
X = [
    [2, 60],
    [3, 65],
    [4, 70],
    [8, 90],
    [9, 95],
    [10, 92]
]

model = KMeans(n_clusters=2, random_state=0)
model.fit(X)

print(model.labels_)
