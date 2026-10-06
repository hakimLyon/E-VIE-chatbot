"""Admin configuration for Aura AI."""

from django.contrib import admin
from .models import ChatMessage, DetectionResult, SentimentAnalysis


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('message', 'created_at')
    search_fields = ('message',)
    list_filter = ('created_at',)
    readonly_fields = ('created_at',)


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ('image', 'created_at')
    list_filter = ('created_at',)
    readonly_fields = ('created_at',)


@admin.register(SentimentAnalysis)
class SentimentAnalysisAdmin(admin.ModelAdmin):
    list_display = ('sentiment', 'score', 'created_at')
    search_fields = ('text',)
    list_filter = ('sentiment', 'created_at')
    readonly_fields = ('created_at',)
