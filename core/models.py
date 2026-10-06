"""Models for Aura AI application."""

from django.db import models


class ChatMessage(models.Model):
    """Store chat messages."""
    message = models.TextField()
    response = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Chat message - {self.created_at}"


class DetectionResult(models.Model):
    """Store object detection results."""
    image = models.ImageField(upload_to='detections/')
    result = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Detection - {self.created_at}"


class SentimentAnalysis(models.Model):
    """Store sentiment analysis results."""
    text = models.TextField()
    sentiment = models.CharField(
        max_length=20,
        choices=[
            ('positive', 'Positive'),
            ('negative', 'Negative'),
            ('neutral', 'Neutral'),
        ]
    )
    score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Sentiment: {self.sentiment} - {self.created_at}"
