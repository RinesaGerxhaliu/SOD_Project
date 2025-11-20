import os
import cv2
import numpy as np
import random

IMAGE_SIZE = (224, 224)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    return img.astype("float32") / 255.0

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMAGE_SIZE)
    mask = mask.astype("float32") / 255.0
    return np.expand_dims(mask, -1)

def augment(img, mask):

    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    if random.random() < 0.3:
        factor = 0.6 + random.random() * 0.8
        img = np.clip(img * factor, 0, 1)

    if random.random() < 0.3:
        h, w = IMAGE_SIZE
        crop = random.randint(10, 25)

        img_c = img[crop:h-crop, crop:w-crop]
        mask_c = mask[crop:h-crop, crop:w-crop]

        img = cv2.resize(img_c, IMAGE_SIZE)
        mask = cv2.resize(mask_c, IMAGE_SIZE)
        mask = np.expand_dims(mask, -1)

    return img, mask

def get_dataset(img_folder, mask_folder, do_augment=False):
    X, y = [], []

    for name in sorted(os.listdir(img_folder)):
        if not name.endswith(".jpg"):
            continue

        img_path = os.path.join(img_folder, name)
        mask_path = os.path.join(mask_folder, name.replace(".jpg", ".png"))

        if not os.path.exists(mask_path):
            continue

        img = load_image(img_path)
        mask = load_mask(mask_path)

        if do_augment:
            img, mask = augment(img, mask)

        if mask.ndim == 2:
            mask = np.expand_dims(mask, -1)

        X.append(img)
        y.append(mask)

    return np.array(X), np.array(y)



