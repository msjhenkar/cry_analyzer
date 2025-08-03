# 👶🎵 Baby Cry Sound Analyzer and Lullaby Generator

This project is an AI-powered system designed to **analyze the emotional tone in a baby's cry** and generate a **soothing lullaby** tailored to the identified emotion. By combining machine learning with audio signal processing, the system offers an intelligent and compassionate solution to support infant care and parenting.

---

## 🚀 Project Overview

The core idea behind this project is to **detect emotions** (like hunger, pain, discomfort, or fear) in a baby's cry using audio classification techniques and respond by generating a calming lullaby based on the detected mood.

### 🧠 Technologies Used
- **Python**
- **Scikit-learn** – For model training and emotion classification  
- **Librosa** – For audio feature extraction (MFCCs, Chroma, Spectral Contrast, etc.)
- **Pandas & NumPy** – Data preprocessing

---

## 🛠️ Features

- 🎤 **Audio Input** – Accepts recorded baby cry samples (.wav format)
- 📊 **Feature Extraction** – Extracts Mel-frequency cepstral coefficients (MFCC), pitch, energy, etc.
- 🧠 **Emotion Detection** – Classifies the emotion using a trained **Random Forest Classifier**
- 🎼 **Lullaby Generator** – Suggests or plays lullabies tuned to the baby's detected emotional state
- 📈 **Accuracy Reports** – Evaluation using metrics like precision, recall, and F1-score

---

## 🧪 Emotion Categories

The classifier is trained to detect the following emotional states:
- **Hunger**
- **Pain**
- **Discomfort**
- **Fear**
- **Neutral / Sleepy**

---


---

## 📈 Model Training

- **Classifier:** Random Forest
- **Training/Test Split:** 80/20
- **Features:** MFCC, Chroma, Spectral Centroid, Zero-Crossing Rate, etc.
- **Performance:** Achieved up to 90% accuracy (varies with dataset quality)

---

## 🎶 Lullaby Generation Logic

Based on the detected emotion, a pre-mapped lullaby is played:
- **Pain →** Soft classical piano lullaby  
- **Hunger →** Gentle melody with rhythmic tones  
- **Discomfort →** White noise or humming sounds  
- **Fear →** Warm, harmonic lullaby to reduce anxiety  

---

## 📋 Installation

```bash
git clone https://github.com/yourusername/baby-cry-analyzer.git
cd baby-cry-analyzer
pip install -r requirements.txt
python main.py
```
---

