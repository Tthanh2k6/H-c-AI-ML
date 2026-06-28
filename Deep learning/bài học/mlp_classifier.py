from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os

os.system('cls')
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlp = MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu',
                    max_iter=500, warm_start=True, random_state=42)


mlp.fit(X_train, y_train)
pre_y = mlp.predict(X_test)
acc = accuracy_score(y_test, pre_y)
print("epoch:", mlp.n_iter_)
print("acc: ", acc)