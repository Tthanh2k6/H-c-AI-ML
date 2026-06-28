import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
os.system('cls')
X = np.array([
    [100, 1, 2],
    [150, 2, 3],
    [200, 2, 4],
    [250, 3, 4],
    [300, 3, 5]
])
y = np.array([10, 15, 20, 25, 30])

model = LinearRegression()
model.fit(X, y)
y_DUDoan = model.predict(X)

print("Hệ số góc: ", model.coef_)
print("Giao điểm: ", model.intercept_)

print("Dự đoán nhà  220m2, 2 tầng, 3 phòng ngủ: ", model.predict([[220, 2, 3]]))

