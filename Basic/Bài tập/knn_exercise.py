import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
X = np.array([
    [150, 6],  # Táo
    [160, 5],  # Táo
    [170, 6],  # Táo
    [140, 7],  # Táo
    [130, 8],  # Táo
    [180, 4],  # Cam
    [190, 3],  # Cam
    [200, 2],  # Cam
    [210, 3],  # Cam
    [220, 1]   # Cam
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0: Táo, 1: Cam

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

print("Quả có 160g và độ ngọt 4 là: ", model.predict([[200, 2]]))

