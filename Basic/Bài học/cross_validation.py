from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

texts = [
    "Tôi ghét sản phẩm này",          # 0
    "Không tốt như mong đợi",         # 0
    "Chất lượng bình thường",         # 1
    "Tạm được",                        # 1
    "Tôi thích sản phẩm này",         # 2
    "Rất đáng tiền",                  # 2
    "Không thể tin được!",            # 2
    "Thất vọng quá",                  # 0
    "Khá ổn",                         # 1
    "Tuyệt vời ông mặt trời"         # 2
]

labels = [0, 0, 1, 1, 2, 2, 2, 0, 1, 2]

model = make_pipeline(
    TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        token_pattern=r'\b\w+\b'
    ),
    MultinomialNB()
)

model.fit(texts, labels)

score = cross_val_score(model, texts, labels, cv = 3, scoring='f1_macro')

print("f1 cho từng fold: ", score)
print("f1 trung bình: ", score.mean())