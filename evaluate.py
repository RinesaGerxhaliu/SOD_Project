import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img.astype("float32") / 255.0


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (224, 224))
    mask = mask.astype("float32") / 255.0   
    return np.expand_dims(mask, -1)

BASE = "/content/drive/MyDrive/SOD_Project/dataset"
test_img_path = f"{BASE}/images/test"
test_mask_path = f"{BASE}/masks/test"

THRESH = 0.5

print("Loading model...")
model = tf.keras.models.load_model("sod_model.h5", compile=False)


files = sorted([f for f in os.listdir(test_img_path) if f.endswith(".jpg")])
print("Num test images:", len(files))


ious, precisions, recalls, f1s, maes = [], [], [], [], []

for f in files:
    img = load_image(os.path.join(test_img_path, f))
    gt  = load_mask(os.path.join(test_mask_path, f.replace(".jpg", ".png")))

    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]

    pred_bin = (pred > THRESH).astype("float32")
    gt_bin   = (gt >= 0.5).astype("float32")

    inter = np.logical_and(gt_bin == 1, pred_bin == 1).sum()
    union = np.logical_or(gt_bin == 1, pred_bin == 1).sum()
    iou = inter / union if union > 0 else 0

    tp = np.logical_and(gt_bin == 1, pred_bin == 1).sum()
    fp = np.logical_and(gt_bin == 0, pred_bin == 1).sum()
    fn = np.logical_and(gt_bin == 1, pred_bin == 0).sum()

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)
    mae       = np.abs(gt_bin - pred_bin).mean()

    ious.append(iou)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    maes.append(mae)

print("\nFINAL TEST METRICS (Old Version - High Scores)")
print(f"IoU:       {np.mean(ious):.6f}")
print(f"Precision: {np.mean(precisions):.6f}")
print(f"Recall:    {np.mean(recalls):.6f}")
print(f"F1-score:  {np.mean(f1s):.6f}")
print(f"MAE:       {np.mean(maes):.6f}")

print("\nShowing 4 example predictions...\n")

for idx, f in enumerate(files[:4]):
    img = load_image(os.path.join(test_img_path, f))
    gt  = load_mask(os.path.join(test_mask_path, f.replace(".jpg", ".png")))
    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    pred_bin = (pred > THRESH).astype("float32")

    plt.figure(figsize=(20,5))
    plt.subplot(1,4,1); plt.imshow(img); plt.axis("off"); plt.title("Image")
    plt.subplot(1,4,2); plt.imshow(gt.squeeze(), cmap="gray"); plt.axis("off"); plt.title("GT Mask (Gray)")
    plt.subplot(1,4,3); plt.imshow(pred_bin.squeeze(), cmap="gray"); plt.axis("off"); plt.title("Prediction")
    plt.subplot(1,4,4)
    plt.imshow(img)
    plt.imshow(pred_bin.squeeze(), cmap="jet", alpha=0.4)
    plt.axis("off"); plt.title("Overlay")
    plt.suptitle(f"Example {idx+1}")
    plt.show()
