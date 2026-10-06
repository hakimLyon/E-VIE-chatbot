"""Forms for Aura AI application."""

from django import forms


class ImageUploadForm(forms.Form):
    """Form for uploading images."""
    image = forms.ImageField(
        label='Select image',
        widget=forms.FileInput(attrs={'accept': 'image/*'})
    )


class SentimentAnalysisForm(forms.Form):
    """Form for sentiment analysis."""
    text = forms.CharField(
        label='Text to analyze',
        widget=forms.Textarea(attrs={'rows': 4, 'placeholder': 'Enter text to analyze...'})
    )


class ChatForm(forms.Form):
    """Form for chatbot."""
    message = forms.CharField(
        label='Message',
        widget=forms.TextInput(attrs={'placeholder': 'Type your message...'})
    )
