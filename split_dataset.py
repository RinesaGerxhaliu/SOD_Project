import os, shutil, random

BASE = "/content/drive/MyDrive/SOD_Project"
IMG = f"{BASE}/raw/images"
MASK = f"{BASE}/raw/masks"
OUT = f"{BASE}/dataset"

os.makedirs(f"{OUT}/images/train", exist_ok=True)
os.makedirs(f"{OUT}/images/val", exist_ok=True)
os.makedirs(f"{OUT}/images/test", exist_ok=True)
os.makedirs(f"{OUT}/masks/train", exist_ok=True)
os.makedirs(f"{OUT}/masks/val", exist_ok=True)
os.makedirs(f"{OUT}/masks/test", exist_ok=True)

files = [f for f in os.listdir(IMG) if f.endswith(".jpg")]
random.shuffle(files)

train, val = int(0.7*len(files)), int(0.85*len(files))

for f, split in [(f,"train") for f in files[:train]] + \
                [(f,"val")   for f in files[train:val]] + \
                [(f,"test")  for f in files[val:]]:
    shutil.copy(f"{IMG}/{f}", f"{OUT}/images/{split}/{f}")
    shutil.copy(f"{MASK}/{f.replace('.jpg','.png')}", f"{OUT}/masks/{split}/{f.replace('.jpg','.png')}")
