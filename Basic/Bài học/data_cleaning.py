import pandas as pd
import os
os.system('cls')
a = pd.read_csv('csdl.csv')

#print(a.isnull().sum()) # kiểm tra xem có bao nhiêu dữ liệu trống

cleaned = pd.DataFrame(a.copy())

cleaned = cleaned[cleaned['tip'] >= 1 ] # xóa hàng có tip nhỏ hơn 1
cleaned['tip_percent'] = cleaned['tip'] / cleaned['total_bill'] * 100
cleaned.rename(columns={'sex': 'gender'}, inplace= True)                    # Đổi tên cột
print("Dòng bị trùng: ", cleaned.duplicated().sum())
print(cleaned) 