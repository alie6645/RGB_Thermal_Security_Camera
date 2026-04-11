import os
from pathlib import Path

import tensorflow as tf
from matplotlib import pyplot as plt

IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

# Tutorial checkpoint dir
CHECKPOINT_DIR = Path("training_checkpoints")
OUT_DIR = Path("demo_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Facades dataset path under ~/.keras/datasets
DATASET_NAME = "facades"
DATASET_ROOT = Path.home() / ".keras" / "datasets" / f"{DATASET_NAME}_extracted"

if (DATASET_ROOT / "train").exists():
    PATH = DATASET_ROOT
elif (DATASET_ROOT / DATASET_NAME / "train").exists():
    PATH = DATASET_ROOT / DATASET_NAME
else:
    raise FileNotFoundError(f"Could not find facades dataset under {DATASET_ROOT}")


def load(image_file):
    image = tf.io.read_file(str(image_file))
    image = tf.io.decode_jpeg(image)

    w = tf.shape(image)[1] // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image, real_image


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )

    x = inputs
    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def denorm(img):
    return img * 0.5 + 0.5


def main():
    latest_ckpt = "/home/donpc/projects/RGB_Thermal_Security_Camera/training_checkpoints/ckpt-5"
    if latest_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found in {CHECKPOINT_DIR.resolve()}"
        )

    print("Using checkpoint:", latest_ckpt)

    generator = Generator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        generator=generator,
    )
    checkpoint.restore(latest_ckpt).expect_partial()

    # Pick one facades test image
    test_files = sorted((PATH / "test").glob("*.jpg"))
    if not test_files:
        test_files = sorted((PATH / "val").glob("*.jpg"))
    if not test_files:
        raise FileNotFoundError("No facades test/val jpg files found")

    image_path = test_files[0]
    print("Using image:", image_path)

    input_image, real_image = load_image_test(image_path)
    prediction = generator(input_image[tf.newaxis, ...], training=False)[0]

    display_list = [input_image, real_image, prediction]
    titles = ["Input", "Ground Truth", "Prediction"]

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(denorm(display_list[i]))
        plt.axis("off")

    out_path = OUT_DIR / "facades_demo_prediction.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("Saved demo image to:", out_path.resolve())


if __name__ == "__main__":
    main()