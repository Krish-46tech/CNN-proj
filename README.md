# 🩺 Pneumonia Detection using CNN & Streamlit

A deep learning–based medical imaging project that detects **Pneumonia from chest X-ray images** using a **Convolutional Neural Network (CNN)**.  
The project includes a **Streamlit web application** for interactive predictions and **Grad-CAM visual explanations** to interpret model decisions.

---

## 🚀 Features

- 🧠 **CNN Model (ResNet18)** using transfer learning
- ⚖️ **Class imbalance handling** (weighted loss + balanced sampling)
- 🔄 **Data augmentation** to improve generalization
- 📊 **Prediction confidence scores**
- 🔍 **Grad-CAM visualization** for explainability
- 🌐 **Interactive Streamlit web app**
- 🍎 Optimized for **Apple Silicon (MPS)**

---

## 🏗️ Project Structure


> ⚠️ **Note:**  
> The `data/` folder and `pneumonia_model.pth` are intentionally ignored in GitHub to keep the repository lightweight.

---

## 📂 Dataset

The model is trained on the **Chest X-Ray Pneumonia Dataset**.

- Source: Kaggle  
- Dataset link:  
  👉 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Dataset classes:
- `NORMAL`
- `PNEUMONIA`

---

## 🧠 Model Architecture

- Backbone: **ResNet18 (pretrained on ImageNet)**
- Final layer modified for **binary classification**
- Loss Function: **Weighted CrossEntropyLoss**
- Optimizer: **Adam**
- Input size: **224 × 224 RGB**

---

## 🎯 Training Strategy

To prevent bias toward Pneumonia class:

- ✔️ Data augmentation (rotation, flip, brightness, affine transforms)
- ✔️ WeightedRandomSampler
- ✔️ Class-weighted loss
- ✔️ Transfer learning for faster convergence

---

## 🌐 Streamlit Application

The Streamlit app provides:

- Image upload interface
- Pneumonia / Normal prediction
- Confidence probabilities
- Grad-CAM heatmap visualization
- Automatic model training if no weights are found

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install torch torchvision streamlit opencv-python scikit-learn tqdm matplotlib
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
cd Project
streamlit run app.py

