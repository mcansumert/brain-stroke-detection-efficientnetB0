# Brain Stroke Detection with EfficientNetB0 🧠🩺  
Brain CT stroke classification using **EfficientNetB0** and **Albumentations**

A deep learning pipeline (TensorFlow + Albumentations) for binary classification  
**Stroke** vs **Normal** on axial brain CT slices.  
Includes 5‑fold stratified cross‑validation, class balancing, and data augmentation.

---

## 🔖 Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Results](#results)
- [Citation](#citation)
- [Future Work](#future-work)

---

## 📂 Dataset <a id="dataset"></a>

```
Brain_Data/
├── Normal/
│   ├── img1.jpg
│   └── ...
└── Stroke/
    ├── imgA.jpg
    └── ...
```

- Images must be JPEG (`.jpg`, `.jpeg`) and can be of any size; they will be resized to **224×224**.
- Each class resides in its own subfolder.

> ⚠️ GitHub repository size is limited to 100MB.
> 
> 📦 [📥 Download sample dataset from Google Drive](https://drive.google.com/file/d/1fQr1KKPXlYzNopLCINxTPkbk7Pll2MEn/view?usp=sharing)

---

## ⚙️ Installation <a id="installation"></a>

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 2. Install requirements
pip install -r requirements.txt
```

`requirements.txt`:

```
tensorflow>=2.20
albumentations
scikit-learn
opencv-python-headless
numpy
```

---

## 🧠 Training <a id="training"></a>

```bash
python train_brain_stroke.py
```

- Model checkpoints will be saved as: `efficientnetb0_foldX.h5`
- At the end of all folds, average metrics will be printed.

---

## 📊 Results <a id="results"></a>

| Fold | Accuracy | Precision | Recall | AUC  |
|------|----------|-----------|--------|------|
| 1    | 0.7367   | 0.6659    | 0.8685 | 0.8296 |
| 2    | 0.7223   | 0.6621    | 0.8204 | 0.8076 |
| 3    | 0.7328   | 0.6678    | 0.8444 | 0.8290 |
| 4    | 0.7275   | 0.6698    | 0.8147 | 0.8361 |
| 5    | 0.7332   | 0.6802    | 0.8034 | 0.8299 |
| **Mean** | **0.7305** | **0.6692** | **0.8303** | **0.8264** |

---

## 📚 Citation <a id="citation"></a>

```bibtex
@software{mert2025stroke,
  author       = {Münevver Cansu Mert},
  title        = {Brain CT Stroke Detection with EfficientNetB0},
  version      = {1.0},
  year         = {2025},
  url          = {https://github.com/mcansumert/brain-stroke-detection-efficientnetB0}
}
```

---

## 💡 Future Work <a id="future-work"></a>

- 🔁 Fine-tune with EfficientNetB3 or B4  
- 🧠 Add Grad-CAM or Layer-wise Relevance Propagation  
- 📱 Convert model to TensorFlow Lite for mobile or embedded deployment  
