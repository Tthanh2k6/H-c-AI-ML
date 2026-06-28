import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os

os.system('cls')
# Dữ liệu đầu vào
X = np.array([[1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5], [6.0]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Tạo và huấn luyện mô hình
model = LogisticRegression()
model.fit(X, y)
y_pre = model.predict(X)
y_pro = model.predict_proba(X)

# Dự đoán và in
print("Xác xuất đậu nếu học 3.2 giờ: ", model.predict_proba([[3.2]])[:,1])
print("Kết quả phân loại: ", model.predict([[3.2]]))

#Biểu đồ

plt.scatter(X, y, color='red', label='Dữ liệu')
plt.plot(X, y_pro, color='blue', label = 'Tỉ lệ')
plt.xlabel("Thời gian học")
plt.ylabel("tỉ lệ đậu")
plt.title("Dự đoán tỉ lệ đậu dựa trên số giờ học")
plt.legend()
plt.show()

