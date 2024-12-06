# 이 코드는 CNN모델을 GAN에서 전이학습하여 사용할 수 있게 사전 학습 후 keras파일로 저장하는 코드임.

import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

# MNIST 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0  # 정규화
x_test = x_test / 255.0  # 정규화
x_train = x_train[..., tf.newaxis]  # 채널 추가
x_test = x_test[..., tf.newaxis]

# 데이터 증강 정의
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
])

# 증강된 데이터를 학습에 사용
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (train_dataset
                 .shuffle(60000)
                 .map(lambda x, y: (data_augmentation(x), y))
                 .batch(32)
                 .prefetch(buffer_size=tf.data.AUTOTUNE))

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

# CNN 모델 정의
def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Learning Rate Scheduler 정의
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 모델 생성 및 컴파일
cnn_model = build_cnn_model()
cnn_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 콜백 정의
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_cnn_model.keras', monitor='val_loss', save_best_only=True
)

# 모델 학습
cnn_model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20,
    callbacks=[early_stopping, model_checkpoint]
)

# 모델 저장
cnn_model.save("mnist_cnn_model.keras")
print("CNN 모델이 'mnist_cnn_model.keras'에 저장되었습니다.")
