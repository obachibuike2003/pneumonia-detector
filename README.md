# 🧠 AI-Powered Pneumonia Detector

This is a deep learning-based web application built with **TensorFlow**, **Keras**, and **Flask** that automatically detects **pneumonia** in chest X-ray images. Users can upload X-ray images through a simple web interface, and the model will analyze them and predict whether the lungs are **normal** or show signs of **pneumonia**.

---

## 🚀 Features

* 🔍 Accurate classification using a trained Convolutional Neural Network (CNN)
* 🖼 Upload chest X-ray images directly from your browser
* 📊 Displays model confidence score
* 🧪 Trained on real-world medical dataset (`chest_xray` from Kaggle)
* 🖥 Powered by Flask for easy deployment

---

## 📁 Folder Structure

```
AI PNEUMONIA DETECTOR/
│
├── pneumonia_classifier.py      # Flask app
├── pneumonia_model.h5           # Trained Keras model (not pushed to GitHub)
├── templates/
│   └── index.html               # Frontend HTML file
├── uploads/                     # Temporary image upload folder (excluded from Git)
├── chest_xray/                  # Dataset (excluded from Git)
├── .gitignore                   # Files/folders excluded from Git
└── requirements.txt             # Required Python packages
```

---

## 📸 How It Works

1. Upload a chest X-ray image from your browser.
2. The image is preprocessed and passed into a trained CNN model.
3. The model returns:

   * 🟢 "Normal" – if lungs are healthy
   * 🔴 "Pneumonia" – if signs of infection are detected
4. A confidence score (in %) is displayed along with the result.

---

## 🧪 Model Training Details

* Model type: CNN with data augmentation
* Input size: 150x150 RGB images
* Layers used: Conv2D, MaxPooling, Flatten, Dense, Dropout
* Activation: ReLU and Sigmoid
* Optimizer: Adam
* Dataset: `chest_xray` (train/val/test split)
* Accuracy: \~87% on test set

---

## ⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/obachibuike2003/pneumonia-detector.git
cd pneumonia-detector

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🧬 Run the App

```bash
python pneumonia_classifier.py
```

Then open your browser and go to:
👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## ⚠️ Note

* The model file `pneumonia_model.h5` is **excluded from GitHub** due to its large size (>100MB).
* You must train the model yourself or request the `.h5` file separately.

---

## 🏥 Use Case

This tool can assist **clinics, hospitals, and health professionals** in **pre-screening** for pneumonia using automated AI support — especially in regions where radiologists may be limited.

---

## 🧑‍💻 Author

**Chibuike Obadiegwu**
[GitHub Profile](https://github.com/obachibuike2003)
📍 Anambra, Nigeria

---

Let me know if you'd like to include screenshots, video demo links, or hosting instructions (e.g. Streamlit, Render, or Flask on VPS).
