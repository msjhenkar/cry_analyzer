# train_emotion_model.py
import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Feature Extraction Function
def extract_features(file_path):
    """
    Extract audio features from a file.
    """
    y, sr = librosa.load(file_path, sr=16000)
    pitch = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    intensity = np.mean(librosa.feature.rms(y=y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    return [pitch, intensity, mfcc]

# Step 2: Collecting Features and Labels
data_dir = 'baby_cries_data'  # This folder should contain subfolders of emotions with audio files.
features = []  # To store features of each audio file
labels = []    # To store emotion labels corresponding to each audio file

# Loop through each emotion folder and each file
for emotion in os.listdir(data_dir):
    emotion_folder = os.path.join(data_dir, emotion)
    if os.path.isdir(emotion_folder):
        for file in os.listdir(emotion_folder):
            file_path = os.path.join(emotion_folder, file)
            # Extract features from the audio file
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(emotion)  # Use the folder name as the label

# Step 3: Convert to numpy arrays
X = np.array(features)  # Features for training
y = np.array(labels)    # Labels for training

# Step 4: Encode the Labels
# LabelEncoder is used to convert text labels (e.g., "hungry", "sleepy") to numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Fit and transform labels

# Step 5: Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save the Model and Label Encoder
joblib.dump(model, 'emotion_model.pkl')  # Save the trained model
joblib.dump(label_encoder, 'label_encoder.pkl')  # Save the label encoder
print("Model and label encoder saved as 'emotion_model.pkl' and 'label_encoder.pkl'")