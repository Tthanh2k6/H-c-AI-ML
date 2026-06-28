import pandas as pd

#data = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
#print(data)

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85, 90, 95]
}
df = pd.DataFrame(data)
print(df, "\n")

print(df.head())         # 5 dòng đầu
print(df['Name'])        # Cột 'Name'
print(df.iloc[1])        # Dòng thứ 2
print(df.describe())     # Thống kê nhanh
print(df[df['Age'] > 28])  # Lọc: tuổi > 28
