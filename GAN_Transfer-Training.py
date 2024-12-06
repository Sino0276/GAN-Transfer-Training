# 이 코드는 사전 학습시킨 CNN모델을 50번째 부터 사용함으로써 생성기(Generator)의 학습성능을 높여봄

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt

# 사용자 지정 숫자를 학습
user_digit = int(input("학습할 숫자를 입력하세요 (0~9): "))

# MNIST 데이터 로드 및 전처리
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train[y_train == user_digit]  # 지정된 숫자만 필터링
x_train = x_train / 255.0  # 정규화
x_train = np.expand_dims(x_train, axis=-1)  # 채널 추가

BUFFER_SIZE = x_train.shape[0]
BATCH_SIZE = min(256, BUFFER_SIZE)

# 데이터셋 생성 및 최적화
dataset = (tf.data.Dataset.from_tensor_slices(x_train)
           .shuffle(BUFFER_SIZE)
           .batch(BATCH_SIZE)
           .prefetch(buffer_size=tf.data.AUTOTUNE))

# 생성자 모델 정의
def build_generator():
    return tf.keras.Sequential([
        layers.Input(shape=(100,)),
        layers.Dense(7 * 7 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])

# 간단한 Discriminator 정의 (초기 훈련용)
def build_simple_discriminator():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 진짜/가짜 판별
    ])
    return model

# 강력한 CNN Discriminator 정의 (훈련 후반부에서 사용)
def build_strong_discriminator(user_digit):
    base_model = models.load_model("best_cnn_model.keras")  # CNN 모델 로드
    base_model.trainable = False  # 하위 계층 고정

    model = models.Sequential([
        base_model,
        layers.Lambda(lambda x: x[:, user_digit:user_digit+1]),  # 특정 숫자의 확률만 추출
        layers.Dense(1, activation='sigmoid')  # 진짜/가짜 판별
    ])
    return model

# 모델 초기화
generator = build_generator()
discriminator = build_simple_discriminator()  # 초기에는 간단한 Discriminator 사용

# 손실 함수 및 옵티마이저
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# 단일 훈련 단계
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        # Discriminator 입력
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

# 이미지 생성 및 저장
def generate_and_save_images(model, epoch, test_input, save_dir="generated_images", user_digit=0):
    save_dir = os.path.join(save_dir, str(user_digit))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0  # [-1, 1] -> [0, 1]
    predictions = 1 - predictions

    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axs.flat):
        ax.imshow(predictions[i, :, :, 0], cmap='gray')
        ax.axis('off')
    plt.suptitle(f"Epoch {epoch}", fontsize=16)
    plt.savefig(f"{save_dir}/generated_epoch_{epoch:03d}.png")
    plt.close(fig)

# 훈련 함수
def train(dataset, epochs, user_digit):
    steps_per_epoch = int(np.ceil(BUFFER_SIZE / BATCH_SIZE))
    print(f"Steps per Epoch: {steps_per_epoch} | Total Samples: {BUFFER_SIZE}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs} 시작")

        # 특정 Epoch 이후에 Discriminator 교체
        if epoch == 50:
            print("Discriminator 교체: 강력한 CNN으로 변경")
            global discriminator
            discriminator = build_strong_discriminator(user_digit)
            discriminator_optimizer.learning_rate = 1e-5  # 학습률 낮춤

        for step, image_batch in enumerate(dataset.take(steps_per_epoch)):
            gen_loss, disc_loss = train_step(image_batch)
            if step % 10 == 0:
                print(f"  Step {step}/{steps_per_epoch}: Generator Loss = {gen_loss:.4f}, Discriminator Loss = {disc_loss:.4f}")

        test_input = tf.random.normal([16, 100])
        generate_and_save_images(generator, epoch + 1, test_input, user_digit=user_digit)
        print(f"Epoch {epoch + 1} 완료\n")

# 훈련 실행
EPOCHS = 200
train(dataset, EPOCHS, user_digit)
