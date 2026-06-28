import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("input.csv")
print(df.head())
print(df.info())
print(df['label'].value_counts())

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = make_pipeline(
    TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        token_pattern=r'\b\w+\b'
    ),
    MultinomialNB()
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
ct = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, zero_division=0)
print(acc,'\n',ct,'\n',cr)