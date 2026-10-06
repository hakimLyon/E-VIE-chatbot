"""tests.py - Test suite for Aura AI"""

from django.test import TestCase, Client
from django.urls import reverse
from core.models import ChatMessage, DetectionResult, SentimentAnalysis


class HomeViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_page_loads(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'home.html')


class ChatbotViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_chatbot_page_loads(self):
        response = self.client.get('/chat/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'chatbot.html')


class ObjectDetectionViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_object_detection_page_loads(self):
        response = self.client.get('/detect/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'object_detection.html')


class SentimentAnalysisViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_sentiment_analysis_page_loads(self):
        response = self.client.get('/sentiment/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'sentiment_analysis.html')


class ChatMessageModelTest(TestCase):
    def test_create_chat_message(self):
        msg = ChatMessage.objects.create(
            message="Hello",
            response="Hi there!"
        )
        self.assertEqual(msg.message, "Hello")
        self.assertEqual(msg.response, "Hi there!")
