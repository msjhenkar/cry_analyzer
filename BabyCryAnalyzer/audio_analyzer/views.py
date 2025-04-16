from django.shortcuts import render
import joblib
import numpy as np
import librosa
from .forms import AudioFileForm
from .models import AudioFile

# Load the trained model and label encoder
emotion_model = joblib.load('emotion_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    pitch = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    intensity = np.mean(librosa.feature.rms(y=y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    return [pitch, intensity, mfcc]

# Generate lullaby path based on emotion
def generate_lullaby(emotion):
    lullaby_files = {
        "hungry": "/media/lullabies/hungry.mp3",
        "sleepy": "/media/lullabies/sleepy.mp3",
        "fussy": "/media/lullabies/fussy.mp3",
        "uncomfortable": "/media/lullabies/uncomfortable.mp3",
        "happy": "/media/lullabies/happy.mp3"
    }
    return lullaby_files.get(emotion, "/media/lullabies/default.mp3")

# Main view to handle file upload and analyze emotion
def analyze_audio(request):
    if request.method == 'POST':
        form = AudioFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file to the model
            audio_instance = form.save()
            audio_path = audio_instance.audio_file.path

            # Extract features from the audio file
            features = extract_features(audio_path)
            
            # Make prediction using the model
            prediction = emotion_model.predict([features])[0]
            emotional_state = label_encoder.inverse_transform([prediction])[0]
            
            # Generate lullaby based on prediction
            lullaby_path = generate_lullaby(emotional_state)

            # Render result
            return render(request, 'audio_analyzer/result.html', {
                'emotional_state': emotional_state,
                'lullaby_path': lullaby_path,
            })
    else:
        form = AudioFileForm()
        
    return render(request, 'audio_analyzer/upload.html', {'form': form})

