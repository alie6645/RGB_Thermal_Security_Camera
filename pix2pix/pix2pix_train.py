import os
from pathlib import Path
import random
import time

import tensorflow as tf
from tensorflow.keras import layers

tf.config.optimizer.set_jit(False)

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("/home/donpc/projects/RGB_Thermal_Security_Camera/training_processed_640x480")

IMG_WIDTH = 640
IMG_HEIGHT = 480

BATCH_SIZE = 1
BUFFER_SIZE = 1000
EPOCHS = 100

LAMBDA = 20.0
LEARNING_RATE = 2e-4
BETA_1 = 0.5

CHECKPOINT_DIR = "./training_checkpoints_rgb_to_thermal_from_scratch_v2"
LOG_DIR = "./logs_rgb_to_thermal_from_scratch_v2"
SAMPLE_DIR = "./samples_rgb_to_thermal_from_scratch_v2"

AUTOTUNE = 4
SHUFFLE_PAIRS = True

# Change this if you want a specific session for preview.
# Set to None to use the first test sample.
PREVIEW_SESSION = "session_20260317_201336"

def gradient_loss(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)

    return (
        tf.reduce_mean(tf.abs(dy_true - dy_pred)) +
        tf.reduce_mean(tf.abs(dx_true - dx_pred))
    )

# ============================================================
# FILE DISCOVERY
# ============================================================

def find_pairs(dataset_root: Path):
    pairs = []

    for pair_dir in dataset_root.rglob("pair_*"):
        if not pair_dir.is_dir():
            continue

        rgb_path = pair_dir / "input_rgb.png"
        thermal_path = pair_dir / "thermal_gray.png"

        if rgb_path.exists() and thermal_path.exists():
            pairs.append((str(rgb_path), str(thermal_path)))

    if SHUFFLE_PAIRS:
        random.shuffle(pairs)

    return pairs


def find_first_preview_pair_in_session(pairs, session_name):
    if session_name is None:
        return None

    matches = []

    for rgb_path, thermal_path in pairs:
        p = Path(rgb_path)
        if p.parent.parent.name == session_name:
            matches.append((rgb_path, thermal_path))

    if not matches:
        return None

    matches.sort(key=lambda x: Path(x[0]).parent.name)
    return matches[0]


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image_pair(rgb_path, thermal_path):
    rgb = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_png(rgb, channels=3)
    rgb = tf.image.convert_image_dtype(rgb, tf.float32)

    thermal = tf.io.read_file(thermal_path)
    thermal = tf.image.decode_png(thermal, channels=1)
    thermal = tf.image.convert_image_dtype(thermal, tf.float32)

    rgb = tf.image.resize(rgb, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    thermal = tf.image.resize(thermal, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)

    rgb = (rgb * 2.0) - 1.0
    thermal = (thermal * 2.0) - 1.0

    return rgb, thermal


def load_train_image(rgb_path, thermal_path):
    rgb, thermal = load_image_pair(rgb_path, thermal_path)
    return rgb, thermal


def load_test_image(rgb_path, thermal_path):
    rgb, thermal = load_image_pair(rgb_path, thermal_path)
    return rgb, thermal


def build_datasets(pair_list, split_ratio=0.9):
    n_total = len(pair_list)
    n_train = max(1, int(n_total * split_ratio))

    train_pairs = pair_list[:n_train]
    test_pairs = pair_list[n_train:] if n_total > 1 else pair_list[:1]

    train_rgb = [p[0] for p in train_pairs]
    train_thermal = [p[1] for p in train_pairs]

    test_rgb = [p[0] for p in test_pairs]
    test_thermal = [p[1] for p in test_pairs]

    train_ds = tf.data.Dataset.from_tensor_slices((train_rgb, train_thermal))
    train_ds = train_ds.shuffle(min(len(train_pairs), BUFFER_SIZE))
    train_ds = train_ds.map(load_train_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_rgb, test_thermal))
    test_ds = test_ds.map(load_test_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.batch(1)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, test_ds, train_pairs, test_pairs


# ============================================================
# MODEL
# ============================================================

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=not apply_batchnorm,
        )
    )

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())
    return result


def Generator():
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = layers.Conv2DTranspose(
        1,
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

        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            x = layers.Resizing(
                height=skip.shape[1],
                width=skip.shape[2],
                interpolation="bilinear",
            )(x)

        x = layers.Concatenate()([x, skip])

    x = last(x)

    if x.shape[1] != IMG_HEIGHT or x.shape[2] != IMG_WIDTH:
        x = layers.Resizing(
            height=IMG_HEIGHT,
            width=IMG_WIDTH,
            interpolation="bilinear",
        )(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name="input_image")
    tar = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 1], name="target_image")

    x = layers.Concatenate()([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(
        512,
        4,
        strides=1,
        kernel_initializer=initializer,
        use_bias=False,
    )(zero_pad1)

    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# ============================================================
# LOSSES / OPTIMIZERS
# ============================================================

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
summary_writer = tf.summary.create_file_writer(LOG_DIR)

global_step = tf.Variable(0, trainable=False, dtype=tf.int64)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    edge_loss = gradient_loss(target, gen_output)

    total_gen_loss = gan_loss + (30.0 * l1_loss) + (0.5 * edge_loss)
    return total_gen_loss, gan_loss, l1_loss, edge_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


# ============================================================
# VISUALIZATION
# ============================================================

def denorm(x):
    return (x + 1.0) / 2.0


def save_sample(epoch, test_input, target, prediction):
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    inp = denorm(test_input[0])
    tar = denorm(target[0])
    pred = denorm(prediction[0])

    inp_vis = tf.image.adjust_gamma(inp, gamma=0.7)
    inp_vis = tf.image.adjust_brightness(inp_vis, delta=0.08)
    inp_vis = tf.clip_by_value(inp_vis, 0.0, 1.0)

    tar = tf.image.grayscale_to_rgb(tar)
    pred = tf.image.grayscale_to_rgb(pred)

    inp_vis = tf.image.convert_image_dtype(inp_vis, tf.uint8, saturate=True)
    tar = tf.image.convert_image_dtype(tar, tf.uint8, saturate=True)
    pred = tf.image.convert_image_dtype(pred, tf.uint8, saturate=True)

    stacked = tf.concat([inp_vis, tar, pred], axis=1)

    out_path = os.path.join(SAMPLE_DIR, f"epoch_{epoch:04d}.png")
    tf.keras.utils.save_img(out_path, stacked)

    tf.keras.utils.save_img(os.path.join(SAMPLE_DIR, f"epoch_{epoch:04d}_rgb.png"), inp_vis)
    tf.keras.utils.save_img(os.path.join(SAMPLE_DIR, f"epoch_{epoch:04d}_pred.png"), pred)


# ============================================================
# TRAIN STEP
# ============================================================

@tf.function(jit_compile=False)
def train_step(input_image, target):
    global_step.assign_add(1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        total_gen_loss, gan_loss, l1_loss, edge_loss = generator_loss(
            disc_generated_output,
            gen_output,
            target,
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", total_gen_loss, step=global_step)
        tf.summary.scalar("gen_gan_loss", gan_loss, step=global_step)
        tf.summary.scalar("gen_l1_loss", l1_loss, step=global_step)
        tf.summary.scalar("disc_loss", disc_loss, step=global_step)

    return total_gen_loss, gan_loss, l1_loss, edge_loss, disc_loss


# ============================================================
# TRAIN LOOP
# ============================================================

def fit(train_ds, test_ds, epochs, preview_pair=None):
    if preview_pair is not None:
        rgb_path, thermal_path = preview_pair
        print("Using fixed preview pair:")
        print("  RGB    :", rgb_path)
        print("  Thermal:", thermal_path)

        rgb, thermal = load_test_image(rgb_path, thermal_path)
        example_input = tf.expand_dims(rgb, axis=0)
        example_target = tf.expand_dims(thermal, axis=0)
    else:
        print("Using first test sample for preview.")
        example_input, example_target = next(iter(test_ds))

    for epoch in range(epochs):
        start = time.time()

        print(f"\nEpoch {epoch + 1}/{epochs}")

        for n, (input_image, target) in train_ds.enumerate():
            total_gen_loss, gan_loss, l1_loss, edge_loss, disc_loss = train_step(input_image, target)

            if int(n) % 50 == 0:
                print(
                f"  step {int(n):04d} | "
                f"gen_total={total_gen_loss.numpy():.4f} | "
                f"gan={gan_loss.numpy():.4f} | "
                f"l1={l1_loss.numpy():.4f} | "
                f"edge={edge_loss.numpy():.4f} | "
                f"disc={disc_loss.numpy():.4f}"
            )

        prediction = generator(example_input, training=False)

        print(
            "pred range:",
            float(tf.reduce_min(prediction)),
            float(tf.reduce_max(prediction)),
            "target range:",
            float(tf.reduce_min(example_target)),
            float(tf.reduce_max(example_target)),
        )

        print(
            "pred mean:",
            float(tf.reduce_mean(prediction)),
            "target mean:",
            float(tf.reduce_mean(example_target)),
        )

        save_sample(epoch + 1, example_input, example_target, prediction)

        if (epoch + 1) % 5 == 0:
            saved_path = checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"Saved checkpoint: {saved_path}")

        print(f"Epoch {epoch + 1} completed in {time.time() - start:.2f} sec")

    saved_path = checkpoint.save(file_prefix=checkpoint_prefix)
    print(f"Final checkpoint saved: {saved_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    pairs = find_pairs(DATASET_ROOT)
    print(f"Found {len(pairs)} valid RGB->thermal pairs")

    if len(pairs) == 0:
        raise RuntimeError(
            f"No valid pairs found under {DATASET_ROOT}. "
            "Expected each pair folder to contain input_rgb.png and thermal_gray.png"
        )

    _, sample_th = load_test_image(*pairs[0])
    print(
        "sample thermal min/max:",
        float(tf.reduce_min(sample_th)),
        float(tf.reduce_max(sample_th)),
    )

    train_ds, test_ds, train_pairs, test_pairs = build_datasets(pairs)

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs:  {len(test_pairs)}")

    preview_pair = find_first_preview_pair_in_session(
        pairs,
        session_name=PREVIEW_SESSION,
    )

    if preview_pair is None:
        print("Requested preview session not found; using first test sample.")

    print("Starting training from scratch. No checkpoint restored.")
    fit(train_ds, test_ds, EPOCHS, preview_pair=preview_pair)


if __name__ == "__main__":
    main()