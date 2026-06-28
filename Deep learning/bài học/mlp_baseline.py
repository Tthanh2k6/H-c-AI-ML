from tensorflow import keras
layers = keras.layers

# 1. Tải dữ liệu
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Chuẩn hóa (đưa pixel về 0..1)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# 3. MLP: phẳng ảnh + vài tầng Dense
model = keras.Sequential([
    layers.Input(shape=(28, 28)),      # ảnh 2D
    layers.Flatten(),                  # 28x28 -> 784
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 lớp
])

# 4. Compile (thiết lập cách học)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Huấn luyện (epochs nhỏ để nhanh)
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_split=0.1,
    verbose=1
)

# 6. Đánh giá
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("MLP test accuracy:", test_acc)