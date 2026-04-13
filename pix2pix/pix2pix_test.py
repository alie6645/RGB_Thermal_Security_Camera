import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers

tf.config.optimizer.set_jit(False)

# ============================================================
# CONFIG
# ============================================================

# Folder you want to run inference on
DATASET_ROOT = Path("/home/donpc/projects/RGB_Thermal_Security_Camera/test_processed_640x480")

# Folder containing the trained checkpoints you want to use
CHECKPOINT_DIR = "./training_checkpoints_rgb_to_thermal_resume_every_epoch"

# Where predicted thermal images will be saved
OUTPUT_DIR = "./inference_outputs"

IMG_WIDTH = 640
IMG_HEIGHT = 480

# ============================================================
# FILE DISCOVERY
# ============================================================

def find_rgb_inputs(dataset_root: Path):
    rgb_files = []

    for pair_dir in dataset_root.rglob("pair_*"):
        if not pair_dir.is_dir():
            continue

        rgb_path = pair_dir / "input_rgb.png"
        if rgb_path.exists():
            rgb_files.append(str(rgb_path))
        else:
            print(f"No input_rgb.png in {pair_dir}")

    rgb_files.sort()
    return rgb_files

# ============================================================
# IMAGE LOADING
# ============================================================

def load_rgb_only(rgb_path):
    rgb = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_png(rgb, channels=3)
    rgb = tf.image.convert_image_dtype(rgb, tf.float32)  # [0,1]
    rgb = tf.image.resize(rgb, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
    rgb = (rgb * 2.0) - 1.0  # normalize to [-1,1]
    return rgb

def denorm(x):
    return (x + 1.0) / 2.0

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
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad1)

    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)

    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

# ============================================================
# CHECKPOINT SETUP
# ============================================================

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

# ============================================================
# SAVE OUTPUTS
# ============================================================
def load_seek_png(seek_path):
    seek = tf.io.read_file(str(seek_path))
    seek = tf.image.decode_png(seek, channels=1)
    seek = tf.image.convert_image_dtype(seek, tf.float32)  # [0,1]
    seek = tf.image.resize(
        seek,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR
    )
    return seek

def save_prediction(rgb_path, prediction):
    p = Path(rgb_path)
    pair_dir = p.parent
    pair_name = pair_dir.name
    session_name = pair_dir.parent.name

    seek_path = pair_dir / "thermal_gray.png"
    out_dir = Path(OUTPUT_DIR) / session_name / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # input rgb for display
    inp = tf.io.read_file(str(rgb_path))
    inp = tf.image.decode_jpeg(inp, channels=3)
    inp = tf.image.convert_image_dtype(inp, tf.float32)
    inp = tf.image.resize(inp, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR)

    # brighten input display a bit like training sample view
    inp_vis = tf.image.adjust_gamma(inp, gamma=0.7)
    inp_vis = tf.image.adjust_brightness(inp_vis, delta=0.08)
    inp_vis = tf.clip_by_value(inp_vis, 0.0, 1.0)

    # ground truth thermal
    if seek_path.exists():
        tar = load_seek_png(seek_path)
        tar_rgb = tf.image.grayscale_to_rgb(tar)
    else:
        print(f"Warning: missing seek.png for {pair_dir}")
        tar_rgb = tf.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)

    # predicted thermal
    pred = denorm(prediction[0])
    pred_rgb = tf.image.grayscale_to_rgb(pred)

    # convert to uint8
    inp_vis_u8 = tf.image.convert_image_dtype(inp_vis, tf.uint8, saturate=True)
    tar_u8 = tf.image.convert_image_dtype(tar_rgb, tf.uint8, saturate=True)
    pred_u8 = tf.image.convert_image_dtype(pred_rgb, tf.uint8, saturate=True)

    # side-by-side image: RGB | GT thermal | Pred thermal
    comparison = tf.concat([inp_vis_u8, tar_u8, pred_u8], axis=1)

    # save side-by-side as the main output
    tf.keras.utils.save_img(str(out_dir / "predicted_thermal.png"), comparison)

    # optional: also save raw prediction alone
    tf.keras.utils.save_img(str(out_dir / "pred_only.png"), pred_u8)

# ============================================================
# INFERENCE LOOP
# ============================================================

def run_inference(rgb_files):
    print(f"Running inference on {len(rgb_files)} RGB inputs...")

    for i, rgb_path in enumerate(rgb_files):
        rgb = load_rgb_only(rgb_path)
        input_tensor = tf.expand_dims(rgb, axis=0)

        prediction = generator(input_tensor, training=False)
        save_prediction(rgb_path, prediction)

        if (i + 1) % 25 == 0 or i == 0:
            print(f"Processed {i + 1}/{len(rgb_files)}")

# ============================================================
# MAIN
# ============================================================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    restore_path = "./training_checkpoints_rgb_to_thermal_resume_every_epoch/ckpt-49"

    if not tf.io.gfile.exists(restore_path + ".index"):
        raise RuntimeError(f"Checkpoint not found: {restore_path}")

    checkpoint.restore(restore_path).expect_partial()
    print(f"Restored checkpoint: {restore_path}")

    rgb_files = find_rgb_inputs(DATASET_ROOT)
    print(f"Found {len(rgb_files)} input_rgb.png files")

    if not rgb_files:
        raise RuntimeError(f"No input_rgb.png files found under {DATASET_ROOT}")

    run_inference(rgb_files)
    print("Done.")

if __name__ == "__main__":
    main()