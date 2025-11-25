import os
import cv2
import numpy as np
import random

IMAGE_SIZE = (224, 224)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Bad image:", path)
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    return img.astype("float32") / 255.0


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Bad mask:", path)
        return None

    mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    mask = (mask / 255.0).astype("float32")

    if mask.ndim == 2:
        mask = np.expand_dims(mask, -1)

    return mask


def augment(img, mask):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < 0.3:
        factor = 0.9 + random.random() * 0.2
        img = np.clip(img * factor, 0, 1)

    if random.random() < 0.3:
        h, w = IMAGE_SIZE
        c = random.randint(5, 12)
        img = img[c:h-c, c:w-c]
        mask = mask[c:h-c, c:w-c]

        img = cv2.resize(img, IMAGE_SIZE)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    if mask.ndim == 2:
        mask = np.expand_dims(mask, -1)

    return img, mask


def get_dataset(img_folder, mask_folder, do_augment=False):
    X, y = [], []

    for name in sorted(os.listdir(img_folder)):
        if not name.endswith(".jpg"):
            continue

        img = load_image(os.path.join(img_folder, name))
        mask = load_mask(os.path.join(mask_folder, name.replace(".jpg", ".png")))

        if img is None or mask is None:
            continue

        if do_augment:
            img, mask = augment(img, mask)

        X.append(img)
        y.append(mask)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)