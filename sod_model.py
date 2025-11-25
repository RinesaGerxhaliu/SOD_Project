import tensorflow as tf
from tensorflow.keras import layers, models

def iou_metric(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - inter
    return (inter + 1e-6) / (union + 1e-6)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)

    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    soft_iou = (inter + 1e-6) / (union + 1e-6)

    return 0.6 * bce + 0.4 * (1 - soft_iou)

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

def build_sod_model(input_shape=(224, 224, 3)):
    inp = layers.Input(input_shape)

    c1 = conv_block(inp, 32); p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 256); p4 = layers.MaxPooling2D()(c4)

    bn = conv_block(p4, 512)
    bn = layers.Dropout(0.25)(bn)

    u1 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    u1 = layers.concatenate([u1, c4])
    c5 = conv_block(u1, 256)

    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c5)
    u2 = layers.concatenate([u2, c3])
    c6 = conv_block(u2, 128)

    u3 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c6)
    u3 = layers.concatenate([u3, c2])
    c7 = conv_block(u3, 64)

    u4 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c7)
    u4 = layers.concatenate([u4, c1])
    c8 = conv_block(u4, 32)

    out = layers.Conv2D(1, 1, activation="sigmoid")(c8)

    return models.Model(inp, out)