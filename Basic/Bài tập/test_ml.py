import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Dữ liệu đầu vào (x: diện tích), đầu ra (y: giá nhà)
x = np.array([30, 40, 50, 60, 70, 80]).reshape(-1, 1)
y = np.array([200, 250, 270, 300, 330, 360])

# Tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(x, y)

# Dự đoán
y_pred = model.predict(x)

# Hiển thị kết quả
plt.scatter(x, y, color='blue')
plt.plot(x, y_pred, color='red')
plt.xlabel("Diện tích (m²)")
plt.ylabel("Giá nhà (triệu)")
plt.title("Linear Regression")
plt.show()

# Đánh giá
print("MSE:", mean_squared_error(y, y_pred))
print("R²:", r2_score(y, y_pred))
