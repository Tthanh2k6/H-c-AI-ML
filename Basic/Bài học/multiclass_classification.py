from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
texts = [
    "I hate this item, it broke immediately.",
    "It’s okay, not what I expected but usable.",
    "Absolutely fantastic! Exceeded my expectations.",
    "This is the worst purchase I’ve made.",
    "Mediocre quality, just average overall.",
    "I'm really happy with this, highly recommend!",
    "It works, nothing special though.",
    "Brilliant design and great performance!"
]

labels = [0, 1, 2, 0, 1, 2, 1, 2]

x_train, x_test, y_train, y_test = train_test_split(texts, labels,test_size=0.2)

model = make_pipeline(
    TfidfVectorizer(
        lowercase= True,
        stop_words='english',
        token_pattern=r'\b\w+\b'
    ),
    MultinomialNB()
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
print("accuracy score: ", accuracy_score(y_test, y_pred))
print("report: \n", classification_report(y_test, y_pred, zero_division = 0))

