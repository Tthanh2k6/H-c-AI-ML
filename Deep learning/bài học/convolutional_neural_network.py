from tensorflow import keras
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test[..., None]/255.0
x_train = x_train[..., None]/255.0

#tầng tích chập: 32 bộ lọc 3x3, relu
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation ='relu', input_shape = (28,28,1)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, (3,3), activation ='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train,
            batch_size = 128,
            epochs = 5,
            validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("accuracy: ", test_acc)