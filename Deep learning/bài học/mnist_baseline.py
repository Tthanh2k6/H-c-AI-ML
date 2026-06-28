from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_digits

Input = load_digits()
X, y = Input.data, Input.target.astype('int')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter = 500,
    solver='sgd',
    batch_size=64,
    random_state=1,
    learning_rate_init=0.01
    # verbose=True
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
