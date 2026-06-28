from tensorflow import keras
layers = keras.layers

# 1. Tải dữ liệu
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Chuẩn hóa: 0-255 → 0-1
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# 3. Xây dựng model
Data_aug = keras.Sequential([
    layers.RandomRotation(0.1),         # xoay 10%
    layers.RandomTranslation(0.1, 0.1), # dịch ngang/dọc 10%
    layers.RandomZoom(0.1)              # phóng to/ thu nhỏ 10%
])
model = keras.Sequential([
    keras.Input(shape=(28,28, 1)),
    layers.Flatten(),    
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

# 4. Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Huấn luyện
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

# 6. Đánh giá
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
