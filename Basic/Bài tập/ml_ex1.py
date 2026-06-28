import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Dữ liệu đầu vào
time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
point = np.array([50, 55, 65, 70, 75, 78, 85, 88, 94])

model = LinearRegression()
model.fit(time, point)
point_Du_Đoan = model.predict(time)

#dự đoán nếu học 6.5 giờ sẽ được mấy điểm
p = model.predict([[6.5]])
print("Nếu học 6.5 giờ sẽ được ", p[0], "  điểm")

#độ phù hợp r2, mse(sai số bình phương)
print("Độ phù hợp:", r2_score(point, point_Du_Đoan))
print("Sai số bình phương(MSE): ", mean_absolute_error(point, point_Du_Đoan))
print("Hệ số góc: ", model.coef_)
print("Giao điểm: ", model.intercept_)

#biểu đồ
plt.scatter(time, point, color='red')
plt.plot(time, point_Du_Đoan, color='blue')
plt.title("biểu đồ dự đoán điểm dựa trên thời gian học")
plt.xlabel("Time")
plt.ylabel("Point")
plt.show()