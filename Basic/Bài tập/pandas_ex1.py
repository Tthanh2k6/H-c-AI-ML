import pandas as pd 
import os
import matplotlib.pyplot as plt

os.system('cls')
'''
a = pd.DataFrame({'Name': ["Thanh", "Bao", "Trung", "Thy", "Hao", "Loc"],
              'Age':  [19, 18, 19, 14, 15, 16],
              'Major': ["T", "L", "H", "S", "S", "D"],
              'GPA': [3.5, 4.0 , 2.0, 3.4, 2.8, 2.5]})

print(a)                    #hiển thị toàn bộ mảng
print(a.head(2))            #hiển thị 2 dòng đầu
print(a['Name'])            #cột 'Name'
print(a[a['GPA'] > 3.0])    #GPA > 3.0
'''
a = pd.read_csv('csdl.csv')

print("Có ", a.shape[1], "dòng dữ liệu")
print("Trung bình tiền típ: ", a['tip'].mean())
print("Số người là nam giới: ", a[a['sex'] == "Male"].shape[0])
print("Bill trên 30: ", a[a['total_bill'] > 30].shape[0])

print("Trung bình tiền tip:\n", a.groupby('sex')['tip'].mean())
print("Trung bình tổng bill theo ngày:\n", a.groupby('day')['total_bill'].mean())

a['total_bill'].hist(bins=20)
plt.title("Tổng hoá đơn")
plt.xlabel("Giá trị")
plt.ylabel("Số lượng")
plt.show()