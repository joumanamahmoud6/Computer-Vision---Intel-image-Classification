# Computer Vision — Multi-Class Image Classification

A complete end-to-end Computer Vision pipeline that classifies natural scene images into one of six categories using a custom CNN and transfer learning with MobileNetV2.

---

## Problem Statement

Given an input image of a natural scene, the system predicts exactly one of six scene categories:
`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`

---

## Dataset

**Intel Image Classification** — Natural Scene Classification

- ~17,000 RGB images across 6 balanced classes
- Images are approximately 150×150 pixels
- Source: [Kaggle — Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

---

## Project Structure

```
├── notebook.ipynb         # Main Colab notebook (all steps)
├── final_model.keras      # Saved final model (MobileNetV2)
├── best_cnn.keras         # Best baseline CNN checkpoint
├── best_tl.keras          # Best transfer learning checkpoint
└── README.md              # This file
```

---

## Notebook Sections

| # | Section |
|---|---------|
| 1 | Project Introduction & Business Problem |
| 2 | Dataset Loading |
| 3 | Dataset Understanding |
| 4 | Image EDA |
| 5 | Preprocessing |
| 6 | Train / Validation / Test Split |
| 7 | Data Augmentation |
| 8 | Baseline CNN (from scratch) |
| 9 | Training Curves |
| 10 | Evaluation & Confusion Matrix |
| 11 | Error Analysis |
| 12 | Transfer Learning (MobileNetV2) |
| 13 | Model Comparison |
| 14 | Prediction on New Images |
| 15 | Final Questions & Conclusion |

---

## Setup & How to Run

### 1. Open in Google Colab

Upload `Computer_Vision.ipynb` to [Google Colab](https://colab.research.google.com/).

### 2. Enable GPU

> Runtime → Change runtime type → T4 GPU → Save

### 3. Download the Dataset via Kaggle API

- Go to [kaggle.com](https://www.kaggle.com) → Account → Create New API Token
- Upload the downloaded `kaggle.json` when prompted by Cell 1
- The notebook will automatically download and unzip the dataset

### 4. Run All Cells

> Runtime → Run all

Cells must be run in order from top to bottom.

---

## Models

### Baseline CNN (from scratch)
- 4 convolutional blocks with filters: 32 → 64 → 128 → 256
- BatchNormalization + Dropout(0.5) for regularization
- GlobalAveragePooling → Dense(256) → Softmax(6)
- Input size: 150×150
- Optimizer: Adam (lr=1e-3)

### Transfer Learning — MobileNetV2
- Pretrained on ImageNet (frozen base initially)
- Custom head: GlobalAveragePooling → Dense(128) → Dropout(0.3) → Softmax(6)
- Phase 1: Train head only (lr=1e-3)
- Phase 2: Fine-tune top 30 layers (lr=1e-5)
- Input size: 224×224

---

## Results

| Model | Input Size | Val Accuracy | Test Accuracy | Weighted F1 |
|-------|-----------|-------------|--------------|-------------|
| Baseline CNN | 150×150 |  0.8575— |  0.8497— | 0.8489— |
| MobileNetV2 (TL) | 224×224 |  0.9248 — | 0.9210— | —0.9205 |


---

## Loading the Saved Model

```python
import tensorflow as tf

model = tf.keras.models.load_model('final_model.keras')
```

### Running a Prediction

```python
from PIL import Image
import numpy as np

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.array(img, dtype=np.float32)
    )
    img_array = np.expand_dims(img_array, axis=0)
    probs = model.predict(img_array)[0]
    pred_class = CLASS_NAMES[np.argmax(probs)]
    confidence = np.max(probs)
    print(f"Predicted: {pred_class} ({confidence:.2%})")
    return pred_class, confidence

predict_image('your_image.jpg')
```

---

## Requirements

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | Model building & training |
| NumPy | Array operations |
| Pandas | Tabular data & comparison |
| Matplotlib | Plotting |
| Seaborn | Statistical visualizations |
| scikit-learn | Metrics & evaluation |
| Pillow | Image loading & processing |
| kaggle | Dataset download |

All libraries are pre-installed in Google Colab. No manual installation needed.

---

## Author

**Joumana Mahmoud**
Faculty of Computers and Data Science — Alexandria University
- GitHub: [github.com/joumanamahmoud6](https://github.com/joumanamahmoud6)
- LinkedIn: [linkedin.com/in/jomana-mahmoud-bb103b410](https://linkedin.com/in/jomana-mahmoud-bb103b410)
