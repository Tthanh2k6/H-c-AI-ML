from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

texts = [
    "I love this product, it works great!",
    "Worst experience ever, totally disappointed.",
    "Very happy with the purchase.",
    "It broke after one day, horrible.",
    "Excellent quality and amazing service!",
    "Terrible support and bad design.",
    "Fast delivery, good value for money.",
    "Not worth the price, very bad.",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = tích cực, 0 = tiêu cực

x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


model = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words='english', token_pattern=r'\b\w+\b'),
    MultinomialNB()
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Độ chính xác:", accuracy_score(y_test, y_pred))

print("dự đoán câu 'I am not happy with the product': ", model.predict(["I am not happy with the product"]))