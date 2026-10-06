# Installation des Modèles IA Open Source

Ce guide explique comment installer et configurer les modèles IA open source pour l'application Made AI.

## 1. Installation des Dépendances

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## 2. Modèles Utilisés

### 2.1 YOLOv5 - Détection d'Objets (Open Source)

**Description**: YOLO (You Only Look Once) est un algorithme de détection d'objets en temps réel.

**Avantages**:
- ✅ 100% Open Source
- ✅ Rapide et efficace
- ✅ Supporte 80 classes d'objets
- ✅ Pas besoin de clé API
- ✅ Fonctionne localement

**Installation automatique**: Le modèle se télécharge automatiquement au premier lancement.

### 2.2 RoBERTa - Analyse de Sentiment (Open Source)

**Description**: Modèle de transformer pré-entraîné pour l'analyse de sentiment.

**Avantages**:
- ✅ Open Source (Hugging Face)
- ✅ Haute précision
- ✅ Support multilingue
- ✅ Gratuit

### 2.3 DialoGPT - Chatbot (Open Source)

**Description**: Modèle de conversation génératif basé sur GPT.

**Avantages**:
- ✅ Open Source (Microsoft)
- ✅ Conversations naturelles
- ✅ Pas de coût par API
- ✅ Fonctionne localement

## 3. Configuration

### 3.1 Variables d'Environnement

Créez ou mettez à jour votre fichier `.env`:

```env
# Configuration Django
DEBUG=True
SECRET_KEY=votre-secret-key

# Optionnel: Clé Google Gemini (fallback)
GOOGLE_API_KEY=votre-cle-gemini-api
```

### 3.2 Configuration des Modèles

Les modèles se configurent automatiquement dans `core/services.py`:

```python
# YOLO Configuration
self.model.conf = 0.25      # Seuil de confiance (25%)
self.model.iou = 0.45       # Seuil IoU (45%)
self.model.max_det = 50      # Détections maximum
```

## 4. Installation Manuelle (Optionnel)

Si vous voulez télécharger les modèles manuellement:

### 4.1 YOLOv5
```bash
# Cloner le dépôt YOLOv5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les poids pré-entraînés
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

### 4.2 Transformers (Hugging Face)
```bash
# Installation des modèles transformers
pip install transformers[torch]
pip install sentencepiece
```

## 5. Test des Modèles

### 5.1 Tester YOLO
```python
import torch
from PIL import Image

# Charger YOLO
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Tester sur une image
results = model('path/to/image.jpg')
results.print()
```

### 5.2 Tester l'Analyse de Sentiment
```python
from transformers import pipeline

# Charger le modèle
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Tester
result = sentiment("I love this product!")
print(result)
```

### 5.3 Tester le Chatbot
```python
from transformers import pipeline

# Charger le modèle
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

# Tester
response = chatbot("Hello, how are you?")
print(response)
```

## 6. Lancement et Test de l'Application

### 6.1 Démarrer le Serveur

```bash
# Activer l'environnement
source venv/bin/activate

# Lancer le serveur de développement
python manage.py runserver 8001
```

### 6.2 Tester les Fonctionnalités

**🌐 Accès à l'application**: `http://127.0.0.1:8001`

#### Test de Détection d'Objets
1. Allez sur `http://127.0.0.1:8001/detect/`
2. Cliquez sur "Choose Image" ou glissez-déposez une image
3. Cliquez sur "Detect Objects"
4. **Résultat attendu**: 
   - Liste des objets détectés avec confiance
   - Boîtes de détection sur l'image
   - Comptage par classe d'objet

#### Test de l'Analyse de Sentiment
1. Allez sur `http://127.0.0.1:8001/sentiment/`
2. Tapez un texte dans le champ (ex: "I love this amazing product!")
3. Cliquez sur "Analyze Sentiment"
4. **Résultat attendu**:
   - Sentiment: positive/negative/neutral
   - Score de confiance: 0.0 à 1.0
   - Explication du résultat

#### Test du Chatbot
1. Allez sur `http://127.0.0.1:8001/chat/`
2. Tapez un message dans le champ (ex: "Hello, how are you?")
3. Cliquez sur le bouton d'envoi
4. **Résultat attendu**:
   - Réponse générée par l'IA
   - Conversation sauvegardée
   - Interface de chat fonctionnelle

#### Test du Floating Chat
1. Sur n'importe quelle page, cliquez sur le bouton de chat flottant (en bas à droite)
2. Le chat s'ouvre en fenêtre popup
3. Testez la conversation
4. **Résultat attendu**:
   - Chat fonctionnel sans recharger la page
   - Synchronisation avec l'API
   - Fermeture automatique

### 6.3 Vérification des Logs

```bash
# Surveiller les logs Django
python manage.py runserver 8001 --verbosity=2

# Logs attendus:
# - "YOLOv5 model loaded successfully"
# - "Detection completed"
# - "Sentiment analysis completed"
```

## 7. Dépannage

### 7.1 Problèmes Communs

**Erreur: "CUDA out of memory"**
```python
# Dans services.py, forcer l'utilisation du CPU
import torch
torch.cuda.is_available = lambda: False
```

**Erreur: "Model not found"**
```bash
# Nettoyer le cache torch
pip cache purge
```

**Performance lente**
```python
# Réduire la taille du modèle
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)  # nano version
```

**YOLO ne se charge pas**
```bash
# Vérifier l'installation
python -c "import torch; print(torch.__version__)"
python -c "import ultralytics; print('Ultralytics OK')"

# Réinstaller si nécessaire
pip uninstall ultralytics
pip install ultralytics==8.0.0
```

### 7.2 Configuration GPU (Optionnel)

Si vous avez une GPU NVIDIA:

```bash
# Installer CUDA Toolkit (selon votre version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Vérifier l'installation
python -c "import torch; print(torch.cuda.is_available())"
```

### 7.3 Tests Unitaires

```python
# tests/test_ai_services.py
import unittest
from core.services import AIService

class TestAIServices(unittest.TestCase):
    def test_yolo_detection(self):
        result = AIService.detect_objects('test_image.jpg')
        self.assertIn('objects', result)
    
    def test_sentiment_analysis(self):
        result = AIService.analyze_sentiment('I love this!')
        self.assertIn('sentiment', result)
    
    def test_chat(self):
        result = AIService.chat('Hello')
        self.assertIn('response', result)
```

Lancer les tests:
```bash
python manage.py test
```

## 8. Monitoring

### 8.1 Logs

Les modèles génèrent des logs dans la console Django:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("YOLOv5 model loaded successfully")
logger.error(f"Detection failed: {e}")
```

### 8.2 Performance

Pour surveiller les performances:

```python
import time

start_time = time.time()
result = model(image_path)
inference_time = time.time() - start_time

print(f"Detection took {inference_time:.2f} seconds")
```

### 8.3 Métriques Attendues

- **YOLO**: < 0.5s par image (CPU), < 0.1s (GPU)
- **Sentiment**: < 1s par analyse
- **Chat**: < 2s par réponse
- **Mémoire**: < 2GB RAM avec YOLO

## 9. Sécurité

- ✅ Pas de données envoyées à des tiers
- ✅ Traitement local des images
- ✅ Pas de coûts par utilisation
- ✅ Respect de la vie privée

## 10. Mises à Jour

Pour mettre à jour les modèles:

```bash
# Mettre à jour les dépendances
pip install --upgrade ultralytics transformers torch

# Les modèles se téléchargeront automatiquement
```

## 11. Test Complet (Checklist)

Avant de mettre en production:

- [ ] YOLO se charge correctement
- [ ] Détection fonctionne sur images test
- [ ] Analyse de sentiment fonctionne
- [ ] Chatbot répond correctement
- [ ] Floating chat fonctionne
- [ ] Logs sans erreurs critiques
- [ ] Performance acceptable
- [ ] Interface responsive

---

**Note**: Les modèles open source fonctionnent hors ligne une fois téléchargés. Aucune connexion Internet n'est requise pour l'inférence.
