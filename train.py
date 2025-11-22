import os
import tensorflow as tf
from data_loader import get_dataset
from sod_model import build_sod_model, combined_loss, iou_metric

BASE = "/content/drive/MyDrive/SOD_Project/dataset"

print("Loading train/val sets...")
X_train, y_train = get_dataset(f"{BASE}/images/train", f"{BASE}/masks/train", do_augment=True)
X_val,   y_val   = get_dataset(f"{BASE}/images/val",   f"{BASE}/masks/val")

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)

model = build_sod_model()
optimizer = tf.keras.optimizers.Adam(1e-3)

model.compile(
    optimizer=optimizer,
    loss=combined_loss,
    metrics=[iou_metric]
)

checkpoint_dir = "/content/drive/MyDrive/SOD_Project/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(1))
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

start_epoch = 1
if manager.latest_checkpoint:
    print("Resuming from:", manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint)
    start_epoch = int(ckpt.epoch.numpy())

class SaveCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        ckpt.epoch.assign(epoch + 1)
        path = manager.save()
        print("Saved checkpoint:", path)

best_model_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_model.h5",
    monitor="val_iou_metric",
    mode="max",
    save_best_only=True,
    verbose=1
)

print("Training started...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    initial_epoch=start_epoch - 1,
    epochs=25,
    batch_size=8,
    callbacks=[best_model_cb, SaveCheckpoint()],
    verbose=1
)

model.save("sod_model.h5")
print("Training finished!")
