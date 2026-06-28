
plt.scatter(X, y, color='red', label='Dữ liệu')
plt.plot(X, y_pro, color='blue', label = 'Tỉ lệ')
plt.xlabel("Thời gian học")
plt.ylabel("tỉ lệ đậu")
plt.title("Dự đoán tỉ lệ đậu dựa trên số giờ học")
plt.legend()
plt.show()

