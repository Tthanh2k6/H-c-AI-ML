from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "Congratulations! You've won a free ticket.",
    "Hi, how are you doing today?",
    "Claim your free prize now by clicking this link.",
    "Are we still on for dinner tonight?",
    "You have been selected for a special reward.",
    "Don't forget to submit your assignment.",
    "Win money instantly! Limited time offer.",
    "See you at the meeting tomorrow.",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham

x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

model = make_pipeline(
    TfidfVectorizer(
        lowercase = True,
        stop_words='english',
        token_pattern=r'\b\w+\b'
    ),
    MultinomialNB()
)
model.fit(x_train, y_train)

y_pre = model.predict(x_test)

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pre))
print("độ chính xác: ", accuracy_score(y_test, y_pre))
print("Recall: ", recall_score(y_test, y_pre, zero_division = 0))
print("Precision: ", precision_score(y_test, y_pre, zero_division=0))
print("f1 score: ", f1_score(y_test, y_pre, zero_division=0))
print("báo cáo tổng quát: \n", classification_report(y_test, y_pre, zero_division=0))
