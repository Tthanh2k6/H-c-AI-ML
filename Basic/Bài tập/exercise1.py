import numpy as np

a = np.array([1, 2, 3]) # mang 1 chieu

b = np.array([[1, 2],
              [3, 4]]) # mang 2 chieu

c = np.array([3, 4, 5])

d = np.array([[3, 6],
             [1, 9]])
print(a + c) # cộng từng phần tử
print(sum(a)) # tổng các phần tử

print(a-c)

print(a*2)

b.T
print(b.T)
print(np.dot(b, d))
