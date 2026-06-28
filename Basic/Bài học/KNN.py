import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Dữ liệu: thời gian học và kết quả thi
X = np.array([[1.0], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5], [6.0]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0: Rớt, 1: Đậu

# Huấn luyện KNN với K=3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Dự đoán và xác suất
print("Dự đoán nếu học 3.2 giờ: ", model.predict([[3.2]]))
print("Tỉ lệ đậu/rớt: ", model.predict_proba([[3.2]]))

# Tạo dữ liệu mịn để vẽ
X_test = np.linspace(1, 6, 200).reshape(-1, 1)
y_prob = model.predict_proba(X_test)

plt.plot(X_test, y_prob[:, 1], color='blue', label='Tỉ lệ đậu')
plt.plot(X_test, y_prob[:, 0], color='green', label='Tỉ lệ rớt')
plt.scatter(X, y, color='red', label='Dữ liệu gốc')
plt.xlabel("Giờ học")
plt.ylabel("Xác suất")
plt.title("Dự đoán kết quả thi bằng KNN")
plt.legend()
plt.grid(True)
plt.show()
