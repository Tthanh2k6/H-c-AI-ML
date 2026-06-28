from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
os.system('cls')
X = [
    [150, 45],
    [160, 55],
    [170, 70],
    [180, 80],
    [190, 90],
    [165, 65],
    [155, 50],
    [175, 75]
]
y = [0, 0, 1, 1, 1, 1, 0, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

model = KNeighborsClassifier(n_neighbors= 3)
scale = StandardScaler()

X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.transform(X_test)
model.fit(X_train_scale, y_train)

y_pred = model.predict(X_test_scale)


acc = accuracy_score(y_test, y_pred)
print("độ chính xác: ", acc)
