import tensorflow as tf
from tensorflow.keras import layers, models

def iou_metric(y_true, y_pred):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    iou = iou_metric(y_true, y_pred)
    return bce + 0.5 * (1 - iou)


def build_sod_model(input_shape=(224, 224, 3)):
    inputs = layers.Input(input_shape)

    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(256, 3, activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(256, 3, activation="relu", padding="same")(c4)
    p4 = layers.MaxPooling2D()(c4)

    bn = layers.Conv2D(512, 3, activation='relu', padding="same")(p4)
    bn = layers.Conv2D(512, 3, activation='relu', padding="same")(bn)

    u1 = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(bn)
    u1 = layers.concatenate([u1, c4])
    c5 = layers.Conv2D(256, 3, activation="relu", padding="same")(u1)
    c5 = layers.Conv2D(256, 3, activation="relu", padding="same")(c5)

    u2 = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(c5)
    u2 = layers.concatenate([u2, c3])
    c6 = layers.Conv2D(128, 3, activation="relu", padding="same")(u2)
    c6 = layers.Conv2D(128, 3, activation="relu", padding="same")(c6)

    u3 = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(c6)
    u3 = layers.concatenate([u3, c2])
    c7 = layers.Conv2D(64, 3, activation="relu", padding="same")(u3)
    c7 = layers.Conv2D(64, 3, activation="relu", padding="same")(c7)

    u4 = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(c7)
    u4 = layers.concatenate([u4, c1])
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(u4)
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(c8)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c8)

    return models.Model(inputs, outputs)