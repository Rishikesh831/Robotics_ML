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
