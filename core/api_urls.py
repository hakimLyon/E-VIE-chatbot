"""URL Configuration for API endpoints."""

from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    path('detect/', views.detect_objects, name='detect_objects'),
    #path('sentiment/', views.analyze_sentiment, name='analyze_sentiment'),
      #path("sentiment/", views.sentiment_analysis, name="sentiment_analysis"),
    path('sentiment/', views.sentiment_api, name='sentiment_api'),
    #path("api/sentiment/", views.sentiment_api, name="sentiment_api"),
    #path('chat/', views.chat, name='chat'),
    path('chat/', views.chat_api, name='chat_api'),
    path('chat/history/', views.chat_history, name='chat_history'),
]
