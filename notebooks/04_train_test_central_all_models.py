# ================================================================
# 🌍 Satellite Landslide Segmentation — 5 U-Net Variants
# ================================================================

import os, numpy as np, tensorflow as tf, rasterio
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from rasterio.windows import Window

# ----------------------------
# CONFIG
# ----------------------------
RAW_IMAGE = "/content/drive/MyDrive/Villangad_2020/data/raw/20241110_053942_45_24f7_3B_AnalyticMS_SR_8b_clip.tif"
RAW_MASK  = "/content/drive/MyDrive/Villangad_2020/data/raw/central_mask.tif"
OUT_DIR   = "/content/drive/MyDrive/Villangad_2020/experiments_all"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 200
LR = 5e-4

# ----------------------------
# STEP 1: TILE IMAGE + MASK
# ----------------------------
def tile_raster_pair(image_path, mask_path, tile_size=IMG_SIZE, stride=None):
    if stride is None:
        stride = tile_size[0]
    imgs, msks = [], []
    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
        for top in range(0, img_src.height - tile_size[0] + 1, stride):
            for left in range(0, img_src.width - tile_size[1] + 1, stride):
                window = Window(left, top, tile_size[1], tile_size[0])
                img = np.moveaxis(img_src.read(window=window), 0, 2).astype(np.float32)
                mask = mask_src.read(1, window=window).astype(np.uint8)
                img = img / (np.max(img) + 1e-8)
                imgs.append(img)
                msks.append(np.expand_dims(mask, -1))
    return np.array(imgs), np.array(msks)

X, Y = tile_raster_pair(RAW_IMAGE, RAW_MASK)
print(f"✅ Tiled: {len(X)} tiles — shape {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ----------------------------
# MODEL COMPONENTS
# ----------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def encoder_block(x, filters):
    c = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(c)
    return c, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    x = layers.concatenate([x, skip])
    x = conv_block(x, filters)
    return x

# ---------------------------------------------------
# RESIDUAL, ATTENTION, DENSE BLOCKS
# ---------------------------------------------------
def residual_block(x, filters):
    shortcut = layers.Conv2D(filters, (1, 1), padding="same")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)

def attention_gate(x, g, inter_channels):
    theta_x = layers.Conv2D(inter_channels, 2, strides=2, padding="same")(x)
    phi_g = layers.Conv2D(inter_channels, 1, padding="same")(g)
    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation("relu")(add)
    psi = layers.Conv2D(1, 1, activation="sigmoid")(act)
    psi_up = layers.UpSampling2D(size=(2,2), interpolation="bilinear")(psi)
    return layers.Multiply()([x, psi_up])

def dense_block(x, filters):
    x1 = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x2 = layers.Conv2D(filters, 3, padding="same", activation="relu")(layers.Concatenate()([x, x1]))
    x3 = layers.Conv2D(filters, 3, padding="same", activation="relu")(layers.Concatenate()([x, x1, x2]))
    return layers.Concatenate()([x, x1, x2, x3])

def multi_scale_conv(x, filters):
    conv3 = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    conv5 = layers.Conv2D(filters, 5, padding="same", activation="relu")(x)
    return layers.Concatenate()([conv3, conv5])

# ---------------------------------------------------
# 5 MODEL DEFINITIONS
# ---------------------------------------------------
def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)
    b = conv_block(p4, 1024)
    d1 = decoder_block(b, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)
    return models.Model(inputs, outputs, name="U-Net")

def build_resunet(input_shape):
    inputs = layers.Input(shape=input_shape)
    c1 = residual_block(inputs, 64); p1 = layers.MaxPooling2D(2)(c1)
    c2 = residual_block(p1, 128); p2 = layers.MaxPooling2D(2)(c2)
    c3 = residual_block(p2, 256); p3 = layers.MaxPooling2D(2)(c3)
    c4 = residual_block(p3, 512); p4 = layers.MaxPooling2D(2)(c4)
    b = residual_block(p4, 1024)
    u1 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(b)
    u1 = layers.Concatenate()([u1, c4]); d1 = residual_block(u1, 512)
    u2 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(d1)
    u2 = layers.Concatenate()([u2, c3]); d2 = residual_block(u2, 256)
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d2)
    u3 = layers.Concatenate()([u3, c2]); d3 = residual_block(u3, 128)
    u4 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    u4 = layers.Concatenate()([u4, c1]); d4 = residual_block(u4, 64)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)
    return models.Model(inputs, outputs, name="ResU-Net")

def build_attnunet(input_shape):
    inputs = layers.Input(shape=input_shape)
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)
    b = conv_block(p4, 1024)
    g1 = layers.Conv2D(512, 1, padding="same")(b)
    att1 = attention_gate(c4, g1, 256)
    u1 = decoder_block(b, att1, 512)
    g2 = layers.Conv2D(256, 1, padding="same")(u1)
    att2 = attention_gate(c3, g2, 128)
    u2 = decoder_block(u1, att2, 256)
    g3 = layers.Conv2D(128, 1, padding="same")(u2)
    att3 = attention_gate(c2, g3, 64)
    u3 = decoder_block(u2, att3, 128)
    u4 = decoder_block(u3, c1, 64)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(u4)
    return models.Model(inputs, outputs, name="Attn-U-Net")

def build_attnresunet(input_shape):
    inputs = layers.Input(shape=input_shape)
    c1 = residual_block(inputs, 64); p1 = layers.MaxPooling2D(2)(c1)
    c2 = residual_block(p1, 128); p2 = layers.MaxPooling2D(2)(c2)
    c3 = residual_block(p2, 256); p3 = layers.MaxPooling2D(2)(c3)
    c4 = residual_block(p3, 512); p4 = layers.MaxPooling2D(2)(c4)
    b = residual_block(p4, 1024)
    g1 = layers.Conv2D(512, 1, padding="same")(b)
    att1 = attention_gate(c4, g1, 256)
    u1 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(b)
    u1 = layers.Concatenate()([u1, att1]); d1 = residual_block(u1, 512)
    g2 = layers.Conv2D(256, 1, padding="same")(d1)
    att2 = attention_gate(c3, g2, 128)
    u2 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(d1)
    u2 = layers.Concatenate()([u2, att2]); d2 = residual_block(u2, 256)
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d2)
    u3 = layers.Concatenate()([u3, c2]); d3 = residual_block(u3, 128)
    u4 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d3)
    u4 = layers.Concatenate()([u4, c1]); d4 = residual_block(u4, 64)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)
    return models.Model(inputs, outputs, name="Attn-ResU-Net")

def build_asdms(input_shape):
    inputs = layers.Input(shape=input_shape)
    e1 = dense_block(inputs, 32); p1 = layers.MaxPooling2D(2)(e1)
    e2 = dense_block(p1, 64); p2 = layers.MaxPooling2D(2)(e2)
    e3 = dense_block(p2, 128); p3 = layers.MaxPooling2D(2)(e3)
    e4 = dense_block(p3, 256); p4 = layers.MaxPooling2D(2)(e4)
    b = multi_scale_conv(p4, 512)
    g1 = layers.Conv2D(256, 1, padding="same")(b)
    att1 = attention_gate(e4, g1, 128)
    u1 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(b)
    u1 = layers.Concatenate()([u1, att1]); d1 = dense_block(u1, 256)
    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d1)
    u2 = layers.Concatenate()([u2, e3]); d2 = dense_block(u2, 128)
    u3 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d2)
    u3 = layers.Concatenate()([u3, e2]); d3 = dense_block(u3, 64)
    u4 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d3)
    u4 = layers.Concatenate()([u4, e1]); d4 = dense_block(u4, 32)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)
    return models.Model(inputs, outputs, name="ASDMS-Attn-U-Net")

# ---------------------------------------------------
# TRAIN ALL
# ---------------------------------------------------
model_builders = [
    build_unet,
    build_resunet,
    build_attnunet,
    build_attnresunet,
    build_asdms,
]

results = {}

for builder in model_builders:
    name = builder.__name__.replace("build_", "")
    print(f"\n🚀 Training {name} ...")
    model = builder(input_shape=X_train.shape[1:])
    model.compile(
        optimizer=optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2)]
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    test_metrics = model.evaluate(X_test, y_test, verbose=0)
    results[name] = {"loss": test_metrics[0], "accuracy": test_metrics[1], "iou": test_metrics[2]}

    model.save(os.path.join(OUT_DIR, f"{name}.keras"))
    print(f"💾 Saved {name} to {OUT_DIR}/{name}.keras")

# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------
print("\n📊 Final IoU Comparison on Test Set:")
for name, res in results.items():
    print(f"{name:<20} IoU={res['iou']:.4f}  Acc={res['accuracy']:.4f}")
