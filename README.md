# 👗 Smart Outfit Advisor

<p align="center">
  <strong>AI-Powered Outfit Analysis & Style Recommendation System</strong>
</p>

<p align="center">
  An intelligent computer vision system that detects clothing items, analyzes their colors, evaluates color harmony, and provides an overall outfit score.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLO-Object%20Detection-green?style=for-the-badge" alt="YOLO">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/AI-Computer%20Vision-purple?style=for-the-badge" alt="AI">
</p>

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Problem Statement](#-problem-statement)
* [Project Objectives](#-project-objectives)
* [Key Features](#-key-features)
* [How It Works](#-how-it-works)
* [System Architecture](#-system-architecture)
* [Supported Clothing Classes](#-supported-clothing-classes)
* [Technology Stack](#-technology-stack)
* [Model Performance](#-model-performance)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Usage](#-usage)
* [Team](#-team)
* [Future Improvements](#-future-improvements)
* [License](#-license)

---

## 🔎 Overview

**Smart Outfit Advisor** is an AI-powered fashion analysis project designed to understand and evaluate clothing outfits from images.

The system combines **Object Detection**, **Computer Vision**, and **Color Analysis** to transform a clothing image into meaningful style insights.

The main pipeline is:

> **Image → Clothing Detection → Clothing Cropping → Color Extraction → Color Harmony Analysis → Outfit Score**

The project is designed to provide an automated way to understand whether the colors and clothing pieces in an outfit work well together.

---

## 💡 Problem Statement

Choosing a well-coordinated outfit can be difficult because it involves multiple visual factors such as:

* Clothing type
* Color combinations
* Color harmony
* Overall visual consistency

Traditional fashion recommendation systems may rely heavily on manually labeled information or predefined rules.

**Smart Outfit Advisor** aims to automate part of this process using computer vision and machine learning.

---

## 🎯 Project Objectives

The project aims to:

* Detect different clothing items automatically.
* Identify the type of each detected clothing item.
* Extract individual clothing regions from the image.
* Analyze the dominant color of each clothing item.
* Represent colors using RGB and HSV color spaces.
* Apply K-Means clustering for dominant color extraction.
* Convert numerical colors into understandable color names.
* Analyze color harmony between clothing items.
* Generate a **Color Harmony Score /10**.
* Provide an overall outfit evaluation.

---

## ✨ Key Features

### 👕 1. Clothing Detection

The system uses a custom-trained **YOLO object detection model** to identify clothing items in an image.

Detected categories include:

* 👜 Bag
* 👗 Dress
* 🎩 Hat
* 🧥 Jacket
* 👖 Pants
* 👚 Shirt
* 👟 Shoe
* 🩳 Shorts
* 👗 Skirt
* 🕶️ Sunglass

Each detected object is returned with its:

* Bounding box
* Class label
* Confidence score

---

### 🎨 2. Clothing Color Analysis

After detecting the clothing items, the system crops each item and analyzes its dominant color.

The color analysis pipeline uses:

* RGB
* HSV
* K-Means Clustering
* Dominant color extraction
* Color naming

This allows the system to transform raw pixel information into understandable colors such as:

> Black, White, Red, Blue, Green, Beige, Brown, etc.

---

### 🌈 3. Color Harmony Analysis

The detected clothing colors are analyzed together to determine how well they work as a combination.

The system evaluates relationships between colors and generates a:

**Color Harmony Score /10**

This score represents how visually compatible the detected outfit colors are.

---

### ⭐ 4. Outfit Evaluation

The final system combines the detected clothing information and color analysis to provide an overall assessment of the outfit.

The goal is to move from:

> **"What clothes are in the image?"**

to:

> **"How well does this outfit work together?"**

---

## ⚙️ How It Works

```text
                    INPUT IMAGE
                         │
                         ▼
              ┌─────────────────────┐
              │   YOLO Detection    │
              │   Clothing Items    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Bounding Boxes    │
              │   + Class Labels    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Crop Clothing     │
              │      Items          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Color Analysis    │
              │ RGB + HSV + KMeans  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Color Name Mapping  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Color Harmony       │
              │     Analysis        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Outfit Score /10   │
              └─────────────────────┘
```

---

## 🏗️ System Architecture

The project is divided into multiple processing stages:

### Stage 1 — Object Detection

The YOLO model receives the input image and detects clothing objects.

**Input:**

```text
Image
```

**Output:**

```text
Class + Bounding Box + Confidence
```

---

### Stage 2 — Clothing Cropping

Each detected clothing item is cropped from the original image.

Example:

```text
Original Image
      │
      ├── Shirt Crop
      ├── Pants Crop
      ├── Shoes Crop
      └── Bag Crop
```

---

### Stage 3 — Dominant Color Extraction

Each clothing crop is processed using **K-Means clustering** to identify the dominant color.

The system works with:

```text
RGB → HSV → K-Means → Dominant Color
```

---

### Stage 4 — Color Naming

The extracted numerical color is mapped to a human-readable color name.

Example:

```text
RGB: (32, 32, 32)
        ↓
    Dark Gray
```

---

### Stage 5 — Color Harmony

The colors of the detected clothing items are analyzed together.

The system evaluates whether the colors are:

* Complementary
* Analogous
* Neutral
* Similar
* Contrasting

The result contributes to the final **Color Harmony Score**.

---

## 👗 Supported Clothing Classes

| Class      | Description           |
| ---------- | --------------------- |
| `bag`      | Bags and handbags     |
| `dress`    | Dresses               |
| `hat`      | Hats and headwear     |
| `jacket`   | Jackets and outerwear |
| `pants`    | Trousers and pants    |
| `shirt`    | Shirts and tops       |
| `shoe`     | Footwear              |
| `shorts`   | Shorts                |
| `skirt`    | Skirts                |
| `sunglass` | Sunglasses            |

---

## 🧠 Technology Stack

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| **Python**       | Core programming language |
| **YOLO**         | Clothing object detection |
| **OpenCV**       | Image processing          |
| **NumPy**        | Numerical operations      |
| **Pandas**       | Data handling             |
| **Scikit-learn** | K-Means clustering        |
| **Matplotlib**   | Visualization             |
| **PyYAML**       | Dataset configuration     |

---

## 📊 Model Performance

The custom YOLO clothing detection model was evaluated on validation and test datasets.

### Validation Results

| Metric    |      Score |
| --------- | ---------: |
| Precision | **76.13%** |
| Recall    | **78.83%** |
| mAP@50    | **80.89%** |
| mAP@50-95 | **54.66%** |

### Test Results

| Metric    |      Score |
| --------- | ---------: |
| Precision | **81.11%** |
| Recall    | **79.92%** |
| mAP@50    | **83.38%** |
| mAP@50-95 | **56.28%** |

These results indicate that the model performs well overall in detecting the supported clothing categories, while some smaller or less represented classes remain more challenging.

---

## 📦 Dataset

The project uses a custom clothing dataset organized for YOLO object detection.

The dataset contains annotated images divided into:

```text
train
valid
test
```

The training dataset contains the following classes:

```text
bag
dress
hat
jacket
pants
shirt
shoe
shorts
skirt
sunglass
```

The class distribution was also analyzed to identify potential imbalance between clothing categories.

---

## 📁 Project Structure

```text
Smart-Outfit-Advisor/
│
├── clothing--1/
│   └── clothing.v1-cloth.yolo26/
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── data.yaml
│
├── cropped_clothes/
│
├── clothing_detection.py
│
├── color_analysis.py
│
├── outfit_analysis.ipynb
│
├── best_yolo26l.pt
│
├── requirements.txt
│
└── README.md
```

> File names may vary depending on the final project organization.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/esraayaser114/Smart-Outfit-Advisor.git
```

```bash
cd Smart-Outfit-Advisor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet:

```bash
pip install ultralytics opencv-python numpy pandas matplotlib scikit-learn pyyaml
```

---

## ▶️ Usage

After installing the required dependencies, provide an image containing an outfit.

The system will:

1. Load the YOLO model.
2. Detect clothing items.
3. Extract bounding boxes.
4. Crop the detected items.
5. Analyze dominant colors.
6. Determine color relationships.
7. Calculate the color harmony score.
8. Produce the final outfit analysis.

Example workflow:

```python
from clothing_detection import detect_clothing

detected_items = detect_clothing(
    image,
    conf_threshold=0.35
)
```

The detected items can then be passed to the color analysis pipeline.

---

## 📈 Dataset Analysis

An important part of the project was analyzing the class distribution before model training.

The training distribution showed that some categories occur more frequently than others.

For example:

```text
shoe       █████████████████████████ 25.17%
shirt      ██████████████████        18.01%
bag        ██████████████            13.92%
jacket     █████████                 9.38%
skirt      ████████                  8.82%
dress      ██████                    6.45%
pants      ██████                    6.08%
shorts     ████                      4.67%
sunglass   ████                      4.20%
hat        ███                       3.31%
```

This analysis helps identify classes that may require additional training data or targeted augmentation.

---

## 🔮 Future Improvements

Possible future improvements include:

* [ ] Improve detection of small objects such as sunglasses.
* [ ] Add more diverse training images.
* [ ] Apply targeted augmentation for minority classes.
* [ ] Improve dominant color extraction for patterned clothing.
* [ ] Add skin-tone and background handling.
* [ ] Add outfit style classification.
* [ ] Add clothing compatibility recommendations.
* [ ] Build a web interface for uploading outfits.
* [ ] Add personalized fashion recommendations.
* [ ] Develop a real-time camera mode.
* [ ] Improve the final outfit scoring algorithm.

---

## 👥 Team

This project was developed collaboratively by:

### **Esraa Yasser**

Project Team Member

### **Sama Islam**

Project Team Member

### **Rawan Tafeesh**

Project Team Member

### **Nicole Nader**

Project Team Member

---

## 🤝 Team Contributions

| Team Member       | Contribution                                   |
| ----------------- | ---------------------------------              |
|**Sama Eslam**     | Clothing dtection                              |
|**Rawan Tafeesh**  | Color Analysis                                 |
|**Nicole Nader**   | Outfit Suitability Evaluation                  |
|**Esraa Yasser**   | Project Development & Integration & Deployment |

---

## 📄 License

This project was developed as an academic / educational project.

If you intend to publish the repository publicly, a specific open-source license such as **MIT License** can be added according to the team's preference.

---

## ⭐ Acknowledgements

This project makes use of modern computer vision and machine learning techniques, particularly object detection and image-based color analysis.

Special thanks to the open-source community and the tools that made this project possible.

---

<p align="center">
  <strong>Smart Outfit Advisor</strong><br>
  <i>See your outfit. Understand your colors. Improve your style.</i>
</p>
