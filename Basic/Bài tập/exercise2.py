import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#a = np.random.randint(100, size=20)
a = np.random.normal(10, 20, size = 20)
print("Danh sach la: ", a)

print("Mean: ", np.mean(a))                     #trung binh
print("Median: ", np.median(a))                 #gia tri o giua
print("Mode: ", stats.mode(a))                  #gia tri xuat hien nhieu nhat
print("Standard Deviation: ", np.std(a))        #độ lệch chuẩn
print("Variance: ", np.var(a))                  #bình phương std(phương sai)


plt.hist(a, bins=100, edgecolor='red')
plt.title("Histogram")
plt.xlabel("Giá trị")
plt.ylabel("Tần suất")
plt.show()
