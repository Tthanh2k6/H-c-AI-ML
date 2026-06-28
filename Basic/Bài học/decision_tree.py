from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

X = [
    [25, 40000, 0],
    [30, 50000, 1],
    [45, 80000, 1],
    [35, 60000, 0],
    [22, 30000, 0],
    [55, 120000, 1],
    [40, 70000, 1],
    [28, 45000, 0]
]
y = ['no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, zero_division=0)
print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)


Du_Doan = model.predict([[35, 70000, 1]])
print("Dự đoán: ", Du_Doan)

plot_tree(
        model, 
        feature_names=["Tuổi", "Thu nhập", "Nhà"],
        class_names=["No", "Yes"],
        filled=True
    )
plt.show()