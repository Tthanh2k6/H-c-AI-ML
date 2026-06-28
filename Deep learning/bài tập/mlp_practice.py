from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import os

os.system('cls')
digit = load_digits()
X, y = digit.data, digit.target

scaler = StandardScaler()
trans_X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(trans_X, y, test_size= 0.2, random_state= 1)

def report_layer(Nlayer):
    model = MLPClassifier(hidden_layer_sizes = (Nlayer), activation = 'relu',
                          max_iter = 500, warm_start=True, random_state=1)
    
    model.fit(X_train, y_train)
    y_pre1 = model.predict(X_test)
    y_pre2 = model.predict(X_train)
    acc1 = accuracy_score(y_test, y_pre1)
    acc2 = accuracy_score(y_train, y_pre2)
    print("n_iter_: ", model.n_iter_)
    print("train acc", acc2)
    print("test acc: ", acc1, end = "\n------------\n") 

report_layer((32,))
report_layer((64, 32))
report_layer((128, 64, 32))
report_layer((256, 128, 64, 32))