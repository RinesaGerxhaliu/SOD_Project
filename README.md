# Salient Object Detection (SOD) - Deep Learning Project

A completed **Salient Object Detection system** built with Python and TensorFlow/Keras.  
It uses a custom **U-Net-inspired CNN encoder–decoder** to detect the most visually important object in an image  
and generate a binary saliency mask.

---

## Setup Instructions

### 1. Create and Activate a Virtual Environment & Clone the Repository

```bash
# Clone the repository
git clone https://github.com/rinesagerxhaliu/SOD_Project.git
cd SOD_Project

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- tensorflow
- numpy
- opencv-python
- matplotlib
- scikit-learn
- Pillow
- tqdm

---

## 3. Dataset Setup

Before training, structure your dataset like this:

```
dataset/
 ├── images/
 │    ├── train/
 │    ├── val/
 │    └── test/
 └── masks/
      ├── train/
      ├── val/
      └── test/
```

The project uses the **ECSSD dataset**.  
Images and masks are automatically resized to **224×224**, normalized to **[0, 1]**, and preprocessed in the code.

---

## 4. Data Loading & Augmentation

The `data_loader.py` module handles:

- Loading images and masks
- Resizing & normalization
- Expanding mask dimensions
- Data augmentations

---

## 5. Model Architecture (U-Net Inspired)

The CNN model is implemented in `sod_model.py`.

Architecture highlights:
- 4-level encoder (Conv2D -> BatchNorm -> ReLU -> MaxPooling)
- Bottleneck with 512 filters + Dropout
- 4-level decoder (ConvTranspose2D + skip connections)
- Final output:
  ```
  Conv2D(1, kernel_size=1, activation="sigmoid")
  ```

---

## 6. Run Training

To start training the SOD model:

```bash
python train.py
```

Training features:
- Adam optimizer (lr=1e-4)
- Validation at the end of each epoch
- ModelCheckpoint
- ReduceLROnPlateau
- EarlyStopping
- Console logging of loss and IoU metrics

**BONUS (Checkpointing):**
- Checkpoint saving (model + optimizer + epoch)
- Resume support

---

## 7. Run Evaluation

After training, evaluate the model on the test set:

```bash
python evaluate.py
```

Metrics computed:

- IoU
- Precision
- Recall
- F1-score
- MAE

Visualization includes for sample test images:

- Input image  
- Ground-truth mask  
- Predicted saliency mask  
- Overlay of prediction on top of the input image

---

## 8. Example Workflow

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

---

## 9. Demo (Interactive Visualization)

A simple interactive demo is included in `demo_notebook.ipynb`.

The demo allows you to:

- Upload any RGB image  
- Run the model in real time  
- View the **input image**, **predicted saliency mask**, and **overlay**  
- See the **inference time per image** (CPU/GPU)

This is useful for quick testing and for the project presentation.

---

## Author

**Rinesa Gerxhaliu**  
Xponian Program – AI Engineering Stream
