import numpy as np
import matplotlib.pyplot as plt

#a = np.random.randint(10, 100, size=100)
a = np.random.normal(loc=50, scale=10, size=1000)
print("Danh sach la: ", a)

print("Mean: ", np.mean(a))                     #trung binh
print("Median: ", np.median(a))                 #gia tri o giua
print("Standard Deviation: ", np.std(a))        #độ lệch chuẩn
print("Variance: ", np.var(a))                  #bình phương std(phương sai)

print("Min: ", end = '')
for _ in range(5):
    print(np.min(a), end = ' ')
    a = np.delete(a, np.argmin(a))

print("\nMax: ", end = '')
for _ in range(5):
    print(np.max(a), end = ' ')
    a = np.delete(a, np.argmax(a))

plt.hist(a, bins=10, edgecolor = 'black')
plt.xlabel("Số")
plt.ylabel("Tần xuất")
plt.show()