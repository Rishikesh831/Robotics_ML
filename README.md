# 🧠 Embedded ML Object Detection (with Streamlit + OpenCV)

This project demonstrates how to train and deploy a lightweight ML model for object detection/recognition.  
The goal: **make it small enough to run on simple microprocessors (like Raspberry Pi)** while still being interactive via a web app.

---

## 🚀 Features
- 📦 Model trained from scratch with sklearn + optimized for embedded compatibility  
- 🎥 Real-time video/image inference using **OpenCV**  
- 🌐 Simple interactive **Streamlit app** for deployment  
- 🖥️ Compatible with low-power devices (Raspberry Pi, Jetson Nano, etc.)

---

## 📂 Project Structure
```bash
.
├── app/                  # Streamlit frontend
│   └── app.py
├── models/               # Trained models (ignored in git, use Drive/S3/HuggingFace)
├── notebooks/            # Jupyter notebooks for training
│   └── Model_Train.ipynb
├── data/                 # Sample datasets
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md

---

## 🚀 Workflow Overview

### 1. **Data Collection (Notebook 1: Gesture.ipynb)**
- Used **Mediapipe** to capture **21 hand landmarks** (x, y) from webcam.
- Built a dataset of gestures: `open_palm`, `fist`, `thumbs_up`.
- Saved as a **CSV file** with shape `(n_samples, 42 features + label)`.

### 2. **Model Training (Notebook 2: Model_Train.ipynb)**
- Preprocessed features (landmark normalization).
- Compared three ML models:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost (final choice)
- Evaluated accuracy using train-test split.
- Achieved **~94% accuracy** consistently across models.
- Saved final trained model as `xgb_model.pkl` using `joblib`.

### 3. **Deployment (Streamlit App)**
- Built a real-time webcam app (`app.py`) using **Streamlit**.
- Extracted landmarks per frame with Mediapipe.
- Passed features to trained model (`xgb_model.pkl`).
- Displayed predictions on live video feed with OpenCV overlay.
- Added a **label map** so predictions show as:
  - `open_palm`
  - `thumbs_up`
  - `fist`

---
<img width="895" height="637" alt="image" src="https://github.com/user-attachments/assets/4be69e54-5e7a-4bf6-8441-36462af0b729" />
<img width="895" height="637" alt="image" src="https://github.com/user-attachments/assets/d894a6da-9713-4d34-b69b-4a6a901a1ff4" />



