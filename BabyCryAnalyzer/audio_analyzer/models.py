# models.py
from django.db import models

class AudioFile(models.Model):
    audio_file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
