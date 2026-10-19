# AI & Machine Learning Notebook

**Học từ dữ liệu nhỏ. Hiểu từng bước. Thử nghiệm bằng code.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)

Kho bài học và bài tập thực hành Machine Learning, từ xử lý dữ liệu bảng đến phân loại, hồi quy và mạng nơ-ron. Các ví dụ được chia thành script nhỏ để đọc, thay đổi tham số và quan sát kết quả trực tiếp.

[Nội dung](#nội-dung) · [Chạy thử](#chạy-thử) · [Lộ trình đọc](#lộ-trình-đọc) · [Cấu trúc](#cấu-trúc)

## Nội dung

| Mảng kiến thức | Nội dung thực hành |
| --- | --- |
| Dữ liệu | pandas, làm sạch, tiền xử lý và chuẩn hóa đặc trưng |
| Học có giám sát | KNN, cây quyết định, hồi quy và phân loại nhiều lớp |
| Đánh giá mô hình | Confusion matrix, cross-validation, các chỉ số phân loại |
| Quy trình huấn luyện | Pipeline kết hợp biến đổi dữ liệu và mô hình |
| Văn bản | TF-IDF và phân loại văn bản |
| Mạng nơ-ron | MLP, nhận dạng chữ số và ví dụ CNN |

## Chạy thử

Cần Python 3. Mở terminal tại thư mục gốc repo, tạo môi trường riêng và cài thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scipy matplotlib scikit-learn tensorflow
python "Basic/Bài học/KNN.py"
```

Trên macOS/Linux, kích hoạt môi trường bằng `source .venv/bin/activate`.

Ví dụ KNN in dự đoán kết quả thi từ số giờ học và mở biểu đồ xác suất. Để thử mạng nơ-ron phân loại chữ số:

```powershell
python "Deep learning/bài học/mnist_classification.py"
```

Script này dùng `load_digits()` của scikit-learn; tên file có chữ `mnist` nhưng dữ liệu thực tế là bộ digits. Các script học sâu khác có thể dùng nguồn dữ liệu và thư viện khác.

### Chạy bài có dữ liệu CSV

Một số bài đọc `csdl.csv` hoặc `input.csv` theo **thư mục làm việc hiện tại**. Với các bài này, chuyển vào `Basic` trước:

```powershell
cd Basic
python "Bài học/data_cleaning.py"
python "Bài học/data_preprocessing.py"
```

## Lộ trình đọc

1. **Làm quen dữ liệu:** đọc `pandas_intro.py`, `data_cleaning.py`, `data_preprocessing.py`.
2. **Xây mô hình đầu tiên:** thử `KNN.py`, `decision_tree.py` và các bài phân loại/hồi quy.
3. **Đánh giá:** thay đổi cách chia dữ liệu, xem confusion matrix và cross-validation.
4. **Ghép quy trình:** đọc `pipeline.py`, thử thay đổi bước chuẩn hóa.
5. **Mạng nơ-ron:** chuyển sang các ví dụ MLP và CNN trong `Deep learning`.

Mỗi lần thử nên ghi lại dữ liệu, tham số, cách chia tập và chỉ số đánh giá để so sánh kết quả có ý nghĩa.

## Cấu trúc

```text
Basic/
├── Bài học/          # Các ví dụ theo chủ đề
├── Bài tập/          # Bài luyện tập và bài toán nhỏ
├── csdl.csv
└── input.csv
Deep learning/
├── bài học/          # MLP, chữ số, CNN
└── bài tập/          # Thực hành mạng nơ-ron
```

Đây là kho học tập, chưa có bộ phụ thuộc khóa phiên bản hay bộ kiểm thử chung. Một số script mở cửa sổ biểu đồ hoặc tải dữ liệu khi chạy; kết quả giữa các lần chạy có thể khác nếu chưa cố định seed.

---

**Trần Trung Thành** · [@Tthanh2k6](https://github.com/Tthanh2k6)
