from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd

def clean_text():
    text = text.lower()  # chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # loại bỏ dấu câu
    text = re.sub(r'\d+', '', text)      # loại bỏ số
    # stopwords có thể dùng danh sách riêng cho tiếng Việt
    return text

documents = [
    "I loved the movie, it was fantastic!",
    "What a terrible movie. I hated it.",
    "It was an average film, not too bad.",
    "Amazing acting and great direction!",
    "The movie was boring and too long."
]

vectorizer = TfidfVectorizer(
    lowercase=True,              # tự chuyển thành chữ thường
    stop_words='english',        # tự loại bỏ stopwords tiếng Anh
    token_pattern=r'\b\w+\b'     # bỏ dấu câu, chỉ giữ từ (regex đơn giản)
)

X = vectorizer.fit_transform(documents)
ds = vectorizer.get_feature_names_out()

print("danh sách các từ: ")
print(ds)

data = pd.DataFrame(X.toarray(), columns=ds)
print("Ma trận: ", data)