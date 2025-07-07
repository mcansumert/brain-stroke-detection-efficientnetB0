#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brain CT Stroke / Normal Sınıflandırması – EfficientNetB0 + Albumentations
Eğitim betiği: 5‑fold stratified cross‑validation, class‑weighting ve erken durdurma içerir.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight
import albumentations as A
import cv2

# --------------------------- Parametreler ---------------------------
DATASET_PATH      = "data/Brain_Data"          # repo içindeki varsayılan konum
IMG_HEIGHT        = 224
IMG_WIDTH         = 224
BATCH_SIZE        = 16
EPOCHS            = 30
N_SPLITS          = 5                          # cross‑validation fold sayısı
# --------------------------------------------------------------------

NORMAL_DIR  = os.path.join(DATASET_PATH, "Normal")
STROKE_DIR  = os.path.join(DATASET_PATH, "Stroke")

albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.2),
])

# --------------------------------------------------------------------
def get_image_paths_and_labels():
    paths, lbls = [], []
    for label, folder in enumerate([NORMAL_DIR, STROKE_DIR]):
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                paths.append(fpath)
                lbls.append(label)
    return np.array(paths), np.array(lbls)

def albumentations_augmentor(image):
    img = image.numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    aug = albumentations_transform(image=img)["image"]
    return aug

def tf_albumentations_augmentor(image, label):
    img = tf.py_function(func=albumentations_augmentor, inp=[image], Tout=tf.uint8)
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    return img, label

def make_generator(paths, labels, mode="train", shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _decode_resize(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.uint8)
        return img, label

    ds = ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)

    if mode == "train":
        ds = ds.map(tf_albumentations_augmentor, num_parallel_calls=tf.data.AUTOTUNE)

    def _preprocess(img, lbl):
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img, lbl

    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ----------------------------- Eğitim -------------------------------
def main():
    img_paths, img_labels = get_image_paths_and_labels()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold, fold_scores = 1, []

    for tr_idx, te_idx in skf.split(img_paths, img_labels):
        print(f"\n{'='*30}\nFold {fold}/{N_SPLITS}\n{'='*30}")

        tr_p, te_p = img_paths[tr_idx], img_paths[te_idx]
        tr_l, te_l = img_labels[tr_idx], img_labels[te_idx]

        tr_p, val_p, tr_l, val_l = train_test_split(
            tr_p, tr_l, test_size=0.2, stratify=tr_l, random_state=42
        )

        tr_gen  = make_generator(tr_p, tr_l, mode="train")
        val_gen = make_generator(val_p, val_l, mode="valid", shuffle=False)
        te_gen  = make_generator(te_p, te_l, mode="test",  shuffle=False)

        cls_w = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(tr_l),
            y=tr_l
        )
        cls_w = dict(enumerate(cls_w))

        base = EfficientNetB0(include_top=False, weights="imagenet",
                              input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        base.trainable = False

        inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy", Precision(), Recall(), AUC()]
        )

        ckpt_path = f"checkpoints/efficientnetb0_fold{fold}.h5"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
            ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
        ]

        model.fit(
            tr_gen, validation_data=val_gen, epochs=EPOCHS,
            class_weight=cls_w, callbacks=callbacks, verbose=2
        )

        loss, acc, prec, rec, auc_ = model.evaluate(te_gen, verbose=0)
        fold_scores.append(dict(loss=loss, accuracy=acc,
                                precision=prec, recall=rec, auc=auc_))
        print(f"Fold {fold} – Test Accuracy: {acc:.4f}, AUC: {auc_:.4f}")
        fold += 1

    # Ortalama metrikler
    mean_scores = {k: np.mean([d[k] for d in fold_scores]) for k in fold_scores[0]}
    print("\n=== Ortalama Sonuçlar ===")
    for k, v in mean_scores.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
