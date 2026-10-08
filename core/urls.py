"""URL Configuration for core app - Template Views."""

from django.urls import path
from . import views


from django.urls import path
from . import views

app_name = 'core'
urlpatterns = [
    path('', views.home, name='home'),
    path('detect/', views.object_detection, name='object_detection'),
     #path('sentiment/', views.sentiment_analysis, name='sentiment_analysis'),
    #path("sentiment/", views.sentiment_analysis, name="sentiment_analysis"),
    path('sentiment/', views.sentiment_page, name='sentiment_analysis'),
    path('chat/', views.chatbot, name='chatbot'),
    path('documents/<str:filename>', views.document_file, name='document_file'),
]



