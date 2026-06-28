from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

texts = [
    "Tôi rất thất vọng về sản phẩm này",
    "Không tốt như mong đợi",
    "Chất lượng bình thường",
    "Cũng tạm được",
    "Tôi thích sản phẩm này",
    "Rất đáng tiền",
    "Tuyệt vời ông mặt trời",
    "Tôi không hài lòng",
    "Khá ổn áp",
    "Cực kỳ tuyệt vời"
]
labels = [0, 0, 1, 1, 2, 2, 2, 0, 1, 2]
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3)


model = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words='english', token_pattern=r'\b\w+\b'),
    MultinomialNB()
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, zero_division=0)
print(acc,"\n", cm,"\n", cr)