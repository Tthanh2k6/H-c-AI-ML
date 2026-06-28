# AI-ML — Học Máy & Học Sâu

> Tổng hợp bài học và bài tập tự thực hành về **Machine Learning** và **Deep Learning** (Python).
> Mỗi file là một chủ đề độc lập, chạy được riêng để minh hoạ một thuật toán hoặc kỹ thuật.

**Tự code 100%, không dùng AI hỗ trợ** — viết tay để hiểu rõ từng bước từ tiền xử lý dữ liệu,
huấn luyện mô hình đến đánh giá kết quả.

---

## 📂 Cấu trúc thư mục

```text
AI-ML/
├── Basic/                       # Machine Learning cơ bản (scikit-learn)
│   ├── Bài học/                 # Code mẫu theo từng chủ đề
│   │   ├── pandas_intro.py              # Làm quen pandas (đọc/lọc/biến đổi dữ liệu)
│   │   ├── data_cleaning.py             # Làm sạch dữ liệu
│   │   ├── data_preprocessing.py        # Tiền xử lý dữ liệu đầu vào
│   │   ├── scaling.py                   # Chuẩn hoá đặc trưng (StandardScaler)
│   │   ├── KNN.py                       # Phân loại K-Nearest Neighbors
│   │   ├── decision_tree.py             # Cây quyết định (Decision Tree)
│   │   ├── multiclass_classification.py # Phân loại nhiều lớp
│   │   ├── confusion_matrix.py          # Ma trận nhầm lẫn & các chỉ số đánh giá
│   │   ├── cross_validation.py          # Kiểm định chéo (cross-validation)
│   │   ├── pipeline.py                  # Gộp tiền xử lý + mô hình bằng Pipeline
│   │   └── tf_idf.py                    # Vector hoá văn bản TF-IDF + Naive Bayes
│   ├── Bài tập/                  # Bài tự luyện
│   │   ├── house_price_prediction.py    # Hồi quy tuyến tính dự đoán giá nhà
│   │   ├── pass_fail.py                 # Phân loại đỗ/trượt
│   │   ├── knn_exercise.py              # Luyện KNN
│   │   └── ...                          # exercise1/2, ml_ex1, pandas_ex1, practice, project, test_ml
│   ├── csdl.csv                  # Dữ liệu mẫu
│   └── input.csv                # Dữ liệu mẫu
│
└── Deep learning/               # Học sâu (TensorFlow / Keras)
    ├── bài học/
    │   ├── mlp_baseline.py              # Mạng perceptron nhiều lớp (MLP) cơ bản
    │   ├── mlp_classifier.py            # MLP phân loại
    │   ├── mnist_baseline.py            # Baseline trên bộ chữ số MNIST
    │   ├── mnist_classification.py      # Phân loại chữ số viết tay MNIST
    │   ├── mnist_mlp.py                 # MNIST bằng MLP
    │   └── convolutional_neural_network.py  # Mạng nơ-ron tích chập (CNN)
    └── bài tập/
        └── mlp_practice.py             # Bài luyện MLP
```

---

## 🛠️ Công nghệ sử dụng

- **Python 3**
- **scikit-learn** — KNN, Decision Tree, Linear/Logistic Regression, Naive Bayes, MLP, pipeline, cross-validation, các chỉ số đánh giá.
- **TensorFlow / Keras** — MLP & CNN cho phần Deep Learning.
- **NumPy**, **pandas**, **SciPy** — xử lý số liệu và dữ liệu bảng.
- **Matplotlib** — trực quan hoá kết quả.

---

## ⚡ Cài đặt & Chạy

```bash
# Cài thư viện cần thiết
pip install scikit-learn tensorflow numpy pandas scipy matplotlib

# Chạy thử một bài bất kỳ
python "Basic/Bài học/KNN.py"
python "Deep learning/bài học/mnist_classification.py"
```

> Mỗi file độc lập, chạy trực tiếp để xem kết quả huấn luyện và biểu đồ. Một số bài đọc dữ liệu
> từ `Basic/csdl.csv` / `Basic/input.csv` — chạy từ thư mục gốc dự án để đường dẫn tương đối khớp.

---

## 📌 Ghi chú

Dự án phục vụ mục đích **tự học**: từ Machine Learning cơ bản (phân loại, hồi quy, đánh giá mô hình)
đến Deep Learning (MLP, CNN, MNIST). Code viết tay từng bước, không dùng AI sinh code.

- **Tác giả**: Trần Trung Thành ([@Tthanh2k6](https://github.com/Tthanh2k6))
