import os
from pathlib import Path
import random
import tensorflow as tf
from tensorflow.keras import layers
import time
import tensorflow as tf

tf.config.optimizer.set_jit(False)
# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("/home/donpc/projects/RGB_Thermal_Security_Camera/processed_640x480")

# expects:
# processed_640x480/
#   session_xxx/
#     pair_xxx/
#       input_rgb.png
#       thermal_gray.png

IMG_WIDTH = 640
IMG_HEIGHT = 480

BATCH_SIZE = 1
BUFFER_SIZE = 1000
EPOCHS = 100

LAMBDA = 20.0
LEARNING_RATE = 2e-4
BETA_1 = 0.5

CHECKPOINT_DIR = "./training_checkpoints_rgb_to_thermal"
LOG_DIR = "./logs_rgb_to_thermal"
SAMPLE_DIR = "./samples_rgb_to_thermal"

AUTOTUNE = tf.data.AUTOTUNE

# Set this False if you want deterministic loading order
SHUFFLE_PAIRS = True

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


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image_pair(rgb_path, thermal_path):
    rgb = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_png(rgb, channels=3)
    rgb = tf.image.convert_image_dtype(rgb, tf.float32)  # [0,1]

    thermal = tf.io.read_file(thermal_path)
    thermal = tf.image.decode_png(thermal, channels=1)
    thermal = tf.image.convert_image_dtype(thermal, tf.float32)  # [0,1]

    rgb = tf.image.resize(rgb, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    thermal = tf.image.resize(thermal, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)

    # Normalize to [-1, 1]
    rgb = (rgb * 2.0) - 1.0
    thermal = (thermal * 2.0) - 1.0

    return rgb, thermal


def random_jitter(rgb, thermal):
    # Upscale slightly, then random crop back
    resize_h = int(IMG_HEIGHT * 1.05)
    resize_w = int(IMG_WIDTH * 1.05)

    rgb = tf.image.resize(rgb, [resize_h, resize_w], method=tf.image.ResizeMethod.BILINEAR)
    thermal = tf.image.resize(thermal, [resize_h, resize_w], method=tf.image.ResizeMethod.BILINEAR)

    stacked = tf.concat([rgb, thermal], axis=-1)
    stacked = tf.image.random_crop(stacked, size=[IMG_HEIGHT, IMG_WIDTH, 4])

    rgb = stacked[:, :, :3]
    thermal = stacked[:, :, 3:]

    if tf.random.uniform(()) > 0.5:
        rgb = tf.image.flip_left_right(rgb)
        thermal = tf.image.flip_left_right(thermal)

    return rgb, thermal


def load_train_image(rgb_path, thermal_path):
    rgb, thermal = load_image_pair(rgb_path, thermal_path)
    # rgb, thermal = random_jitter(rgb, thermal)  # disabled for now
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
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_rgb, test_thermal))
    test_ds = test_ds.map(load_test_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(1)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, test_ds, train_pairs, test_pairs


# ============================================================
# MODEL BUILDING BLOCKS
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
            use_bias=not apply_batchnorm
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
            use_bias=False
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
        downsample(64, 4, apply_batchnorm=False),  # (240, 320, 64)
        downsample(128, 4),                        # (120, 160, 128)
        downsample(256, 4),                        # (60, 80, 256)
        downsample(512, 4),                        # (30, 40, 512)
        downsample(512, 4),                        # (15, 20, 512)
        downsample(512, 4),                        # (8, 10, 512)
        downsample(512, 4),                        # (4, 5, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),      # (8, 10, 512)
        upsample(512, 4, apply_dropout=True),      # (16, 20, 512)
        upsample(512, 4, apply_dropout=True),      # (32, 40, 512)
        upsample(256, 4),                          # (64, 80, 256)
        upsample(128, 4),                          # (128, 160, 128)
        upsample(64, 4),                           # (256, 320, 64)
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = layers.Conv2DTranspose(
        1, 4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh"
    )  # (480, 640, 1)

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
                interpolation="bilinear"
            )(x)

        x = layers.Concatenate()([x, skip])

    x = last(x)

    if x.shape[1] != IMG_HEIGHT or x.shape[2] != IMG_WIDTH:
        x = layers.Resizing(
            height=IMG_HEIGHT,
            width=IMG_WIDTH,
            interpolation="bilinear"
        )(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name="input_image")
    tar = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 1], name="target_image")

    x = layers.Concatenate()([inp, tar])  # (480, 640, 4)

    down1 = downsample(64, 4, False)(x)    # (240, 320, 64)
    down2 = downsample(128, 4)(down1)      # (120, 160, 128)
    down3 = downsample(256, 4)(down2)      # (60, 80, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(
        512, 4, strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(zero_pad1)

    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)

    last = layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer
    )(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# ============================================================
# LOSSES / OPTIMIZERS
# ============================================================

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
summary_writer = tf.summary.create_file_writer(LOG_DIR)


# ============================================================
# VISUALIZATION
# ============================================================

def denorm(x):
    return (x + 1.0) / 2.0


def save_sample(epoch, test_input, target, prediction):
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # keep these as tensors until all tf.image ops are done
    inp = denorm(test_input[0])     # [0,1]
    tar = denorm(target[0])         # [0,1]
    pred = denorm(prediction[0])    # [0,1]

    # --- display-only brightening for the RGB panel ---
    # Lift dark areas a bit so people in shadow are easier to see.
    # Adjust these if needed.
    inp_vis = tf.image.adjust_gamma(inp, gamma=0.7)
    inp_vis = tf.image.adjust_brightness(inp_vis, delta=0.08)
    inp_vis = tf.clip_by_value(inp_vis, 0.0, 1.0)

    # convert grayscale target/prediction to RGB while still tensors
    if inp_vis.shape[-1] == 1:
        inp_vis = tf.image.grayscale_to_rgb(inp_vis)
    tar = tf.image.grayscale_to_rgb(tar)
    pred = tf.image.grayscale_to_rgb(pred)

    # convert to uint8 tensors
    inp_vis = tf.image.convert_image_dtype(inp_vis, tf.uint8, saturate=True)
    tar = tf.image.convert_image_dtype(tar, tf.uint8, saturate=True)
    pred = tf.image.convert_image_dtype(pred, tf.uint8, saturate=True)

    # stitch horizontally: brightened RGB | target thermal | predicted thermal
    stacked = tf.concat([inp_vis, tar, pred], axis=1)

    out_path = os.path.join(SAMPLE_DIR, f"epoch_{epoch:04d}.png")
    tf.keras.utils.save_img(out_path, stacked)

    # optional: save the brightened RGB alone too
    rgb_out_path = os.path.join(SAMPLE_DIR, f"epoch_{epoch:04d}_rgb.png")
    tf.keras.utils.save_img(rgb_out_path, inp_vis)

    # optional: save prediction alone too
    pred_out_path = os.path.join(SAMPLE_DIR, f"epoch_{epoch:04d}_pred.png")
    tf.keras.utils.save_img(pred_out_path, pred)


# ============================================================
# TRAIN STEP
# ============================================================

@tf.function(jit_compile=False)
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        total_gen_loss, gan_loss, l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", total_gen_loss, step=epoch)
        tf.summary.scalar("gen_gan_loss", gan_loss, step=epoch)
        tf.summary.scalar("gen_l1_loss", l1_loss, step=epoch)
        tf.summary.scalar("disc_loss", disc_loss, step=epoch)

    return total_gen_loss, gan_loss, l1_loss, disc_loss


# ============================================================
# TRAIN LOOP
# ============================================================

def fit(train_ds, test_ds, epochs, preview_pair=None):
    if preview_pair is not None:
        rgb_path, thermal_path = preview_pair
        rgb, thermal = load_test_image(rgb_path, thermal_path)
        example_input = tf.expand_dims(rgb, axis=0)
        example_target = tf.expand_dims(thermal, axis=0)
    else:
        example_input, example_target = next(iter(test_ds))

    for epoch in range(epochs):
        start = time.time()

        print(f"\nEpoch {epoch + 1}/{epochs}")
        step_count = 0

        for n, (input_image, target) in train_ds.enumerate():
            total_gen_loss, gan_loss, l1_loss, disc_loss = train_step(
                input_image, target, epoch
            )
            step_count += 1

            if int(n) % 50 == 0:
                print(
                    f"  step {int(n):04d} | "
                    f"gen_total={total_gen_loss.numpy():.4f} | "
                    f"gan={gan_loss.numpy():.4f} | "
                    f"l1={l1_loss.numpy():.4f} | "
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

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Epoch {epoch + 1} completed in {time.time() - start:.2f} sec")

    checkpoint.save(file_prefix=checkpoint_prefix)


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
            f"Expected each pair folder to contain input_rgb.png and thermal_gray.png"
        )

    sample_rgb, sample_th = load_test_image(*pairs[0])

    print(
        "sample thermal min/max:",
        float(tf.reduce_min(sample_th)),
        float(tf.reduce_max(sample_th))
    )

    train_ds, test_ds, train_pairs, test_pairs = build_datasets(pairs)

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs:  {len(test_pairs)}")

    # restore latest checkpoint if present
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest:
        checkpoint.restore(latest)
        print(f"Restored checkpoint: {latest}")

    preview_pair = None

    for rgb_path, thermal_path in pairs:
        if "pair_001" in rgb_path:   # change this to your dark sample
            preview_pair = (rgb_path, thermal_path)
            break

    if preview_pair is None:
        print("Warning: specific preview pair not found, using first test sample.")
        
    fit(train_ds, test_ds, EPOCHS, preview_pair=preview_pair)


if __name__ == "__main__":
    main()