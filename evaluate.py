import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_image, load_mask

IMAGE_SIZE = (224, 224)
BASE = "/content/drive/MyDrive/SOD_Project/dataset"

test_img_path = f"{BASE}/images/test"
test_mask_path = f"{BASE}/masks/test"

THRESH = 0.5

model = tf.keras.models.load_model("sod_model.h5", compile=False)

files = sorted([f for f in os.listdir(test_img_path) if f.endswith(".jpg")])

ious = []
precisions = []
recalls = []
f1s = []
maes = []

def show_result(image, gt, pred, title):
    plt.figure(figsize=(20,5))
    plt.subplot(1,4,1); plt.imshow(image); plt.axis("off"); plt.title("Image")
    plt.subplot(1,4,2); plt.imshow(gt.squeeze(), cmap="gray"); plt.axis("off"); plt.title("Ground Truth")
    plt.subplot(1,4,3); plt.imshow(pred.squeeze(), cmap="gray"); plt.axis("off"); plt.title("Prediction")
    plt.subplot(1,4,4); plt.imshow(image); plt.imshow(pred.squeeze(), cmap="jet", alpha=0.4); plt.axis("off"); plt.title("Overlay")
    plt.suptitle(title)
    plt.show()

for idx, f in enumerate(files):
    img = load_image(os.path.join(test_img_path, f))
    gt  = load_mask(os.path.join(test_mask_path, f.replace(".jpg", ".png")))

    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    pred_bin = (pred >= THRESH).astype("float32")

    inter = np.logical_and(gt==1, pred_bin==1).sum()
    union = np.logical_or(gt==1, pred_bin==1).sum()
    iou = inter / union if union > 0 else 0

    tp = np.logical_and(gt==1, pred_bin==1).sum()
    fp = np.logical_and(gt==0, pred_bin==1).sum()
    fn = np.logical_and(gt==1, pred_bin==0).sum()

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)
    mae       = np.abs(gt - pred_bin).mean()

    ious.append(iou)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    maes.append(mae)

    if idx < 4:
        show_result(img, gt, pred_bin, f"Example {idx+1}")

print("\nFINAL TEST METRICS")
print(f"IoU:       {np.mean(ious):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall:    {np.mean(recalls):.4f}")
print(f"F1-score:  {np.mean(f1s):.4f}")
print(f"MAE:       {np.mean(maes):.4f}")
