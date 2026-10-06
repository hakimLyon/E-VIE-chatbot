# Documentation Backend - Made AI Application

## Architecture Générale

Cette documentation décrit l'architecture backend de l'application Made AI avec Django.

## Structure du Projet

```
aura-ai-django/
├── aura_project/          # Configuration principale
│   ├── __init__.py
│   ├── settings.py        # Paramètres Django
│   ├── urls.py           # URLs principales
│   └── wsgi.py
├── core/                 # Application principale
│   ├── models.py         # Modèles de données
│   ├── views.py          # Vues et logique métier
│   ├── urls.py           # URLs de l'application
│   ├── forms.py          # Formulaires
│   └── admin.py          # Administration Django
├── templates/            # Templates HTML
├── static/              # Fichiers statiques
├── media/               # Fichiers uploadés
└── requirements.txt     # Dépendances Python
```

## Modèles de Données à Implémenter

### 1. Modèle ChatMessage
```python
# core/models.py
from django.db import models
from django.contrib.auth.models import User

class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    session_id = models.CharField(max_length=100, null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Message from {self.user or 'Anonymous'} at {self.created_at}"
```

### 2. Modèle UploadedImage
```python
class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return self.filename
```

### 3. Modèle DetectionResult
```python
class DetectionResult(models.Model):
    uploaded_image = models.ForeignKey(UploadedImage, on_delete=models.CASCADE)
    detected_objects = models.JSONField()  # Stocke les objets détectés
    confidence_scores = models.JSONField()  # Scores de confiance
    processed_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-processed_at']
    
    def __str__(self):
        return f"Detection for {self.uploaded_image.filename}"
```

### 4. Modèle SentimentAnalysis
```python
class SentimentAnalysis(models.Model):
    text = models.TextField()
    sentiment = models.CharField(max_length=50)  # positive, negative, neutral
    confidence = models.FloatField()
    analyzed_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    class Meta:
        ordering = ['-analyzed_at']
    
    def __str__(self):
        return f"Sentiment: {self.sentiment} ({self.confidence:.2f})"
```

## Vues à Implémenter

### 1. API Chat
```python
# core/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    try:
        data = json.loads(request.body)
        message = data.get('message', '')
        
        # TODO: Intégrer avec un vrai modèle de chatbot
        # Pour l'instant, réponse simple
        response_text = f"Vous avez dit: '{message}'. Je traite votre demande..."
        
        # Sauvegarder en base de données
        ChatMessage.objects.create(
            message=message,
            response=response_text,
            user=request.user if request.user.is_authenticated else None,
            session_id=request.session.session_key
        )
        
        return JsonResponse({'response': response_text})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

### 2. Vue Object Detection
```python
@csrf_exempt
@require_http_methods(["POST"])
def object_detection_api(request):
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image uploaded'}, status=400)
        
        uploaded_image = request.FILES['image']
        
        # Sauvegarder l'image
        image_obj = UploadedImage.objects.create(
            image=uploaded_image,
            filename=uploaded_image.name,
            user=request.user if request.user.is_authenticated else None
        )
        
        # TODO: Intégrer avec un modèle de détection d'objets
        # Pour l'instant, résultat simulé
        detected_objects = [
            {'label': 'person', 'confidence': 0.95},
            {'label': 'car', 'confidence': 0.87}
        ]
        
        # Sauvegarder les résultats
        DetectionResult.objects.create(
            uploaded_image=image_obj,
            detected_objects=detected_objects,
            confidence_scores=[obj['confidence'] for obj in detected_objects]
        )
        
        return JsonResponse({
            'success': True,
            'objects': detected_objects,
            'image_url': image_obj.image.url
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

### 3. Vue Sentiment Analysis
```python
@csrf_exempt
@require_http_methods(["POST"])
def sentiment_analysis_api(request):
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        
        # TODO: Intégrer avec un modèle d'analyse de sentiment
        # Pour l'instant, analyse simple basée sur mots-clés
        positive_words = ['bon', 'excellent', 'super', 'génial', 'heureux']
        negative_words = ['mauvais', 'terrible', 'horrible', 'triste', 'colère']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.5 + (positive_count * 0.1))
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.5 + (negative_count * 0.1))
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        # Sauvegarder en base de données
        SentimentAnalysis.objects.create(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            user=request.user if request.user.is_authenticated else None
        )
        
        return JsonResponse({
            'sentiment': sentiment,
            'confidence': confidence,
            'text': text
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

## Configuration des URLs

```python
# core/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/detect/', views.object_detection_api, name='object_detection_api'),
    path('api/sentiment/', views.sentiment_analysis_api, name='sentiment_analysis_api'),
    path('', views.home, name='home'),
    path('chat/', views.chatbot, name='chatbot'),
    path('detect/', views.object_detection, name='object_detection'),
    path('sentiment/', views.sentiment_analysis, name='sentiment_analysis'),
]
```

## Migration de la Base de Données

Après avoir défini les modèles, exécutez:

```bash
python manage.py makemigrations
python manage.py migrate
```

## Dépendances Recommandées

```txt
# requirements.txt
Django>=4.2.0
Pillow>=10.0.0
numpy>=1.24.0
opencv-python>=4.8.0
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.3.0
python-decouple>=3.8
```

## Intégration avec des Modèles IA

### 1. Chatbot avec HuggingFace
```python
from transformers import pipeline

class ChatbotService:
    def __init__(self):
        self.chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
    
    def get_response(self, message):
        response = self.chatbot(message)
        return response[0]['generated_text']
```

### 2. Détection d'Objets avec YOLO
```python
import cv2
import numpy as np

class ObjectDetectionService:
    def __init__(self):
        # Charger le modèle YOLO pré-entraîné
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        
    def detect_objects(self, image_path):
        image = cv2.imread(image_path)
        # Logique de détection d'objets
        # Retourner la liste des objets détectés
        pass
```

### 3. Analyse de Sentiment avec Transformers
```python
from transformers import pipeline

class SentimentService:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis")
    
    def analyze(self, text):
        result = self.analyzer(text)
        return {
            'sentiment': result[0]['label'],
            'confidence': result[0]['score']
        }
```

## Sécurité

1. **Validation des entrées**: Toujours valider et nettoyer les données entrantes
2. **CSRF Protection**: Utiliser les décorateurs CSRF de Django
3. **Rate Limiting**: Implémenter des limites de taux pour les API
4. **File Upload Security**: Valider les types de fichiers et tailles

## Déploiement

1. **Variables d'environnement**: Utiliser python-decouple pour les configurations
2. **Base de données**: Configurer PostgreSQL pour la production
3. **Static Files**: Configurer S3 ou équivalent pour les fichiers statiques
4. **Media Files**: Configurer un service de stockage pour les uploads

## Monitoring

1. **Logging**: Implémenter un logging structuré
2. **Performance**: Surveiller les temps de réponse des API
3. **Errors**: Configurer des alertes pour les erreurs critiques

---

Cette documentation sert de guide pour implémenter la logique backend complète de l'application Made AI.
