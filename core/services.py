"""AI services for Aura AI application."""

import json
import cv2
import numpy as np
import torch
from PIL import Image
import os
import google.generativeai as genai
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Configure Google API with key from settings
if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY)


class ObjectDetectionService:
    """YOLO-based object detection service (Open Source)."""
    
    def __init__(self):
        """Initialize YOLO model."""
        try:
            # Using YOLOv5 from ultralytics (open source)
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.conf = 0.25  # Confidence threshold
            self.model.iou = 0.45  # IoU threshold
            self.model.max_det = 50  # Maximum detections
            logger.info("YOLOv5 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def detect_objects(self, image_path: str):
        """
        Detect objects in an image using YOLO.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing detection results
        """
        try:
            if self.model is None:
                return self._fallback_to_gemini(image_path)
            
            # Perform inference
            results = self.model(image_path)
            
            # Parse results
            detections = []
            objects_detected = []
            
            for i, det in enumerate(results.xyxy[0]):  # xyxy format
                if len(det) == 0:
                    continue
                    
                # Extract detection info
                x1, y1, x2, y2, conf, cls = det.tolist()
                
                # Get class name
                class_name = self.model.names[int(cls)]
                
                detection = {
                    'id': i,
                    'class': class_name,
                    'confidence': round(conf, 3),
                    'bbox': {
                        'x1': round(x1, 2),
                        'y1': round(y1, 2),
                        'x2': round(x2, 2),
                        'y2': round(y2, 2),
                        'width': round(x2 - x1, 2),
                        'height': round(y2 - y1, 2)
                    },
                    'center': {
                        'x': round((x1 + x2) / 2, 2),
                        'y': round((y1 + y2) / 2, 2)
                    }
                }
                
                detections.append(detection)
                objects_detected.append(class_name)
            
            # Get image dimensions
            img = Image.open(image_path)
            width, height = img.size
            
            # Count objects by class
            object_counts = {}
            for obj in objects_detected:
                object_counts[obj] = object_counts.get(obj, 0) + 1
            
            return {
                'success': True,
                'objects': detections,
                'object_counts': object_counts,
                'total_objects': len(detections),
                'image_size': {
                    'width': width,
                    'height': height
                },
                'model_info': {
                    'name': 'YOLOv5s (Open Source)',
                    'classes': len(self.model.names),
                    'confidence_threshold': self.model.conf
                }
            }
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return self._fallback_to_gemini(image_path)
    
    def _fallback_to_gemini(self, image_path: str):
        """Fallback to Gemini when YOLO is not available."""
        try:
            if not settings.GOOGLE_API_KEY:
                return {"error": "Both YOLO and Gemini are not available", "objects": []}
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            with open(image_path, 'rb') as img_file:
                image_data = img_file.read()

            response = model.generate_content([
                "Identify and list all objects you see in this image. Provide results as JSON with 'objects' array containing object names and their approximate positions.",
                {"mime_type": "image/jpeg", "data": image_data}
            ])

            # Parse response
            try:
                result = json.loads(response.text)
                if not isinstance(result.get('objects'), list):
                    result = {"objects": [response.text]}
            except json.JSONDecodeError:
                result = {"objects": [response.text]}

            return result
        except Exception as e:
            return {"error": f"All detection methods failed: {str(e)}", "objects": []}


class AIService:
    """Main AI service interface."""

    @staticmethod
    def check_api_key():
        """Check if API key is configured."""
        if not settings.GOOGLE_API_KEY:
            return False
        return True

    @staticmethod
    def detect_objects(image_path):
        """Detect objects in image using YOLO (Open Source)."""
        try:
            if not image_path or not os.path.exists(image_path):
                return {"error": "Image file not found", "objects": []}
            
            # Try YOLO first (Open Source)
            detector = ObjectDetectionService()
            result = detector.detect_objects(image_path)
            
            return result
            
        except Exception as e:
            return {"error": f"Detection failed: {str(e)}", "objects": []}

    @staticmethod
    def analyze_sentiment(text):
        """Analyze sentiment of text."""
        try:
            if not AIService.check_api_key():
                return {"error": "GOOGLE_API_KEY is not configured. Please set it in .env file", "sentiment": "neutral", "score": 0.0, "explanation": ""}
            
            if not text or len(text.strip()) == 0:
                return {"error": "No text provided", "sentiment": "neutral", "score": 0.0, "explanation": ""}
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""Analyze sentiment of this text and respond with JSON format:
            {{"sentiment": "positive/negative/neutral", "score": 0.0 to 1.0, "explanation": "brief explanation"}}
            
            Text: {text}"""

            response = model.generate_content(prompt)

            try:
                result = json.loads(response.text)
                if 'sentiment' not in result:
                    result['sentiment'] = 'neutral'
                if 'score' not in result:
                    result['score'] = 0.5
                if 'explanation' not in result:
                    result['explanation'] = response.text
            except json.JSONDecodeError:
                result = {
                    "sentiment": "neutral",
                    "score": 0.5,
                    "explanation": response.text
                }

            return result
        except Exception as e:
            return {"error": f"Sentiment analysis failed: {str(e)}", "sentiment": "neutral", "score": 0.0, "explanation": ""}

    @staticmethod
    def chat(message):
        """Chat with AI."""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(message)
            return {"response": response.text}
        except Exception as e:
            return {"error": str(e), "response": ""}
