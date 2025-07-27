# Pneumonia Detection from Chest X ray Images using CNN and VGG16 Transfer Learning

This project is focused on detecting **pneumonia** from chest X-ray images using deep learning. It utilizes both a custom Convolutional Neural Network (CNN) and a transfer learning approach with VGG16. The objective is to classify images into two categories: **Normal** and **Pneumonia**. The dataset used for this task is the widely recognized Kaggle Chest X-ray dataset.

---

## 📁 Dataset

- **Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Structure**:
Each folder contains:
- `NORMAL/`: Chest X-rays without pneumonia
- `PNEUMONIA/`: Chest X-rays with pneumonia

---

## 🧠 Models Implemented

### 1. Custom CNN
- Built from scratch using `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.
- Includes dropout to reduce overfitting.

### 2. Transfer Learning with VGG16
- Pretrained VGG16 model used as a feature extractor (ImageNet weights).
- Custom dense layers added on top.

---
## ⚙️ Key Features

- Data augmentation using `ImageDataGenerator` (rotation, zoom, flipping).
- Proper use of **train**, **validation**, and **test** sets.
- Visualizations for EDA and evaluation:
- Sample X-rays from both classes
- Class distribution
- Training history (accuracy and loss curves)
- Confusion matrix
- Misclassified and correctly classified images

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**

---

## 🔧 Requirements

- Python 3.7+
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn

> ✅ This project was developed and run entirely on **Kaggle Notebooks**, and is compatible with Google Colab.

---

## 📈 Results

| Model       | Validation Accuracy | Remarks                      |
|-------------|---------------------|-------------------------------|
| Custom CNN  | ~69%                | Basic model with good generalization |
| VGG16       | ~81%                | Performs better with transfer learning |


---



## 🧩 Possible Improvements

- Use more advanced architectures (ResNet, EfficientNet).
- Introduce GAN-generated images for augmentation.
- Add early stopping and model checkpointing.
- Perform hyperparameter tuning.

---


> ⚠️ **Disclaimer**: This project is for educational purposes only and is not intended for clinical or diagnostic use.

