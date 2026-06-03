# 💊 Pill AI Pro

> **AI-Powered Multimodal Pill Identification System using Deep Learning, OCR, and Pharmaceutical Metadata Integration**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-orange)
![Status](https://img.shields.io/badge/Status-Research%20Project-purple)

---

## 📖 Project Overview

Pill AI Pro is an AI-powered pill identification system that combines Computer Vision, Optical Character Recognition (OCR), and pharmaceutical metadata retrieval to identify tablets and capsules from uploaded images.

The system utilizes a fine-tuned MobileNetV3-Large Convolutional Neural Network (CNN) alongside EasyOCR and a Decision-Level Fusion mechanism to improve identification reliability and provide detailed medication information.

Unlike traditional pill identification systems that rely heavily on manual user inputs such as shape, color, and imprint text, Pill AI Pro performs automated image-based recognition and intelligent metadata enrichment.

---

## 🌍 Motivation & Real-World Relevance

Although Pill AI Pro was developed as an academic dissertation project, its inspiration originates from real-world challenges faced by healthcare professionals, pharmaceutical regulators, forensic investigators, and law-enforcement agencies.

In many situations, unidentified tablets and capsules are recovered from:

* Pharmaceutical investigations
* Drug trafficking raids
* Undercover operations
* Hospital emergency admissions
* Poison control cases
* Customs inspections
* Illegal drug manufacturing investigations
* Drug abuse and overdose investigations

The identification process often requires access to specialized pharmaceutical databases and expert knowledge.

Pill AI Pro aims to serve as an AI-assisted decision-support tool by narrowing down possible matches and providing a starting point for further expert verification.

### Potential Beneficiaries

* 🏥 Hospitals and Emergency Departments
* 💊 Pharmacists and Pharmaceutical Researchers
* 🧪 Forensic Laboratories
* 🚔 Narcotics and Drug Investigation Units
* 🏛 Regulatory Authorities
* 🎓 Academic Researchers
* 📚 Educational Institutions
* ⚕ Poison Control Centers

---

## 🎯 Project Objectives

The primary objectives of this project are:

* Identify pills directly from uploaded images.
* Fine-tune a deep learning model for pharmaceutical image classification.
* Extract imprint and dosage information using OCR.
* Improve prediction confidence through Decision-Level Fusion.
* Retrieve drug metadata using NIH RxNorm APIs.
* Provide visual analytics and performance monitoring.
* Demonstrate deployment of AI-powered healthcare applications on consumer-grade hardware.

---

## 🔍 Scope of Project

### Included

* ✅ Tablet identification
* ✅ Capsule identification
* ✅ Deep learning-based image classification
* ✅ OCR-assisted imprint recognition
* ✅ Drug metadata retrieval
* ✅ User feedback collection
* ✅ Prediction confidence analysis
* ✅ Performance analytics dashboard

### Excluded

* ❌ Injectable medications
* ❌ Liquid medications
* ❌ Clinical diagnosis
* ❌ Prescription recommendation
* ❌ Medical decision-making
* ❌ International drug databases (current version)

The current implementation primarily focuses on medications available within the United States pharmaceutical ecosystem due to the availability of structured public datasets.

---

## 🚀 Key Features

### 🤖 AI-Based Pill Recognition

* MobileNetV3-Large architecture
* Transfer Learning from ImageNet
* Fine-tuned pharmaceutical image classification
* Multi-class pill recognition

### 🔤 OCR Integration

* EasyOCR-powered imprint detection
* Dosage extraction
* Pill marking recognition
* Confidence-assisted validation

### 🧠 Decision-Level Fusion

Combines:

* CNN prediction confidence
* OCR-detected text information

to improve prediction reliability and ranking accuracy.

### 🌐 NIH RxNorm Integration

Retrieves:

* Drug names
* Generic names
* Dosage information
* Drug forms
* Brand information
* Pharmaceutical metadata

### 📊 Analytics Dashboard

Provides:

* Prediction confidence visualization
* RMSE tracking
* User validation statistics
* Model performance insights

### 📝 Feedback System

Allows users to:

* Validate predictions
* Report incorrect classifications
* Support future model improvement

---

## 🏗 System Architecture

```text
User Upload
      │
      ▼
Flask Web Application
      │
      ▼
Image Preprocessing
      │
      ▼
MobileNetV3-Large CNN
      │
      ├────────► Top Predictions
      │
      ▼
EasyOCR
      │
      ├────────► Imprint Detection
      │
      ▼
Decision-Level Fusion
      │
      ▼
RxNorm API Lookup
      │
      ▼
Final Prediction Results
      │
      ▼
Analytics Dashboard
```

---

## 📂 Project Structure

```text
Pill-AI-Pro/
│
├── app.py
├── train.py
├── analytics_utils.py
├── logic_utils.py
│
├── model/
│   ├── pill_model.pth
│   └── class_mapping.json
│
├── dataset/
│   └── ePillID_data/
│
├── static/
│   ├── uploads/
│   ├── charts/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   ├── results.html
│   └── dashboard.html
│
├── tests/
│   └── test_model_integrity.py
│
├── user_feedback.csv
├── requirements.txt
└── README.md
```

---

## 🧰 Technology Stack

### Programming Language

* Python 3.10+

### Machine Learning

* PyTorch
* Torchvision
* MobileNetV3-Large

### OCR

* EasyOCR

### Web Development

* Flask
* HTML5
* CSS3
* JavaScript
* Bootstrap 5

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Image Processing

* Pillow (PIL)

### APIs

* NIH RxNorm API

### Data Storage

* JSON
* CSV

### Version Control

* Git
* GitHub

---

## 💻 Development Environment

| Component        | Details                 |
| ---------------- | ----------------------- |
| Operating System | Windows 11 Pro (64-bit) |
| IDE              | Visual Studio Code      |
| Language         | Python 3.10+            |
| Processor        | Intel Core i7           |
| RAM              | 16 GB                   |
| Storage          | 512 GB SSD              |
| GPU              | Intel Iris Xe Graphics  |

---

## 📦 Dataset Information

### Dataset Used

#### ePillID Benchmark Dataset

The primary dataset used for model training and evaluation.

| Property      | Value                   |
| ------------- | ----------------------- |
| Total Classes | 4,902                   |
| Total Images  | 13,532                  |
| Source        | NIH / NLM               |
| Type          | Tablet & Capsule Images |
| License       | Public Domain           |

Dataset Repository:

https://github.com/usuyama/ePillID-benchmark

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Pill-AI-Pro.git
cd Pill-AI-Pro
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 🏋️ Model Training

To retrain the model:

```bash
python train.py
```

Generated artifacts:

```text
model/
├── pill_model.pth
└── class_mapping.json
```

---

## 🧪 Testing

Run integrity tests:

```bash
python tests/test_model_integrity.py
```

The test suite verifies:

* Model availability
* Weight loading
* Class mapping consistency
* Prediction pipeline integrity
* Output validation

---

## 📈 Evaluation Summary

| Metric                 | Result      |
| ---------------------- | ----------- |
| Training Images        | 13,532      |
| Classes                | 4,902       |
| Epochs                 | 15          |
| Final Loss             | 0.9277      |
| Training Time          | 421 Minutes |
| Top-1 Accuracy         | ~45%        |
| Top-3 Accuracy         | ~70%        |
| OCR Match Success Rate | ~81.8%      |

---

## 🇮🇳 Collaboration & Future Research

One of the biggest limitations in pharmaceutical AI research is the lack of publicly available and standardized datasets from regions such as India.

The current version primarily focuses on U.S. pharmaceutical data because structured datasets are readily available through public sources.

### Vision for an Indian Pill Dataset

The long-term goal is to develop an open and research-friendly dataset containing:

* Indian tablets
* Indian capsules
* Imprint information
* Dosage details
* Manufacturer information
* Packaging metadata
* Regulatory references

Such a dataset could significantly improve AI-powered pill identification systems for:

* Indian hospitals
* CDSCO and regulatory agencies
* Pharmaceutical manufacturers
* Poison control centers
* Research institutions
* Law-enforcement agencies
* Drug enforcement and narcotics investigations

### 🤝 We Welcome Contributions

Contributions are welcome from:

* AI/ML Researchers
* Pharmacists
* Medical Professionals
* Pharmaceutical Manufacturers
* Government Agencies
* Academic Institutions
* Open Source Contributors

Areas of collaboration include:

* Dataset collection
* Annotation and labeling
* OCR improvements
* Vision Transformer implementations
* Explainable AI (XAI)
* Mobile deployment
* Cloud deployment
* Federated learning
* Drug metadata integration

If you are interested in collaborating, feel free to:

* Open an Issue
* Submit a Pull Request
* Share datasets
* Suggest improvements
* Extend the research

Together, we can help build one of the first open-source AI-assisted pill identification systems tailored to the Indian pharmaceutical ecosystem.

---

## 🔮 Future Enhancements

* Vision Transformer (ViT) models
* Larger datasets
* GPU optimization
* Mobile application
* Android deployment
* iOS deployment
* Real-time camera recognition
* Multi-language OCR
* Cloud deployment
* Explainable AI support
* International pharmaceutical datasets

---

## ⚠️ Disclaimer

Pill AI Pro is intended for:

* Academic Research
* Educational Purposes
* Proof-of-Concept Demonstrations
* Pharmaceutical Research

The system is **NOT** a certified medical device and must not be used as a substitute for professional medical advice, diagnosis, treatment, or regulatory decisions.

All predictions generated by the system should be independently verified using certified pharmaceutical databases, pharmacists, healthcare professionals, or relevant authorities.

---

## 📚 References & Resources

### Datasets

* ePillID Benchmark Dataset
  https://github.com/usuyama/ePillID-benchmark

* National Library of Medicine (NLM)
  https://www.nlm.nih.gov/

### Pharmaceutical Resources

* NIH RxNorm API
  https://lhncbc.nlm.nih.gov/RxNav/

* RxNav API Documentation
  https://lhncbc.nlm.nih.gov/RxNav/APIs/

* DailyMed
  https://dailymed.nlm.nih.gov/

### Machine Learning Resources

* PyTorch
  https://pytorch.org/

* Torchvision
  https://pytorch.org/vision/stable/

* MobileNetV3 Research Paper
  https://arxiv.org/abs/1905.02244

### OCR Resources

* EasyOCR
  https://github.com/JaidedAI/EasyOCR

* OpenCV
  https://opencv.org/

### Web Framework

* Flask
  https://flask.palletsprojects.com/

* Bootstrap
  https://getbootstrap.com/

### Healthcare & Regulatory Organizations

* National Institutes of Health (NIH)
  https://www.nih.gov/

* National Library of Medicine (NLM)
  https://www.nlm.nih.gov/

* World Health Organization (WHO)
  https://www.who.int/

* Central Drugs Standard Control Organization (CDSCO – India)
  https://cdsco.gov.in/

* Indian Pharmacopoeia Commission (IPC)
  https://ipc.gov.in/

---

## 👨‍💻 Author

**Vinit Shah**

M.Sc. Computer Applications (Data Science)

Symbiosis Institute of Computer Studies and Research (SICSR)

Symbiosis International University (Deemed)

Academic Year 2025–2026

---

## ⭐ Support the Project

If you found this project useful or would like to contribute to future research, please consider:

* ⭐ Starring the repository
* 🍴 Forking the project
* 🛠 Submitting improvements
* 🤝 Collaborating on pharmaceutical datasets
* 📢 Sharing the project with researchers and healthcare professionals
