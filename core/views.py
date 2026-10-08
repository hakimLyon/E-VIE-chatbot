"""Views for Aura AI application."""

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import ChatMessage, DetectionResult, SentimentAnalysis
from .services.rag.catalog import documents
from .services.rag.rag_chat import SERVICE_BUSY_REPLY, conversation_language
from .services.rag.retrieval import DOCS_PATH
#from .services import AIService
import json


# Template Views
@require_http_methods(["GET"])
def home(request):
    """Home page view."""
    return render(request, 'home.html')


@require_http_methods(["GET"])
def object_detection(request):
    """Object detection page view."""
    return render(request, 'object_detection.html')


@require_http_methods(["GET"])
def sentiment_analysis(request):
    """Sentiment analysis page view."""
    return render(request, 'sentiment_analysis.html')


@require_http_methods(["GET"])
def chatbot(request):
    """Chatbot page view. Opening the page starts a new conversation: the page shows no
    earlier messages, so the assistant must not answer from them either."""
    request.session.pop("chat_history", None)
    return render(request, 'chatbot.html', {'documents': documents()})


@require_http_methods(["GET"])
def document_file(request, filename):
    """A PDF of the knowledge base, so citations can open it at the cited page."""
    path = DOCS_PATH / filename
    if path.suffix != ".pdf" or path.name != filename or not path.is_file():
        raise Http404("Document not found")
    response = FileResponse(open(path, "rb"), content_type="application/pdf", filename=filename)
    response["Cache-Control"] = "public, max-age=86400"
    return response


# API Views
@csrf_exempt
@api_view(['POST'])
def detect_objects(request):
    """API endpoint for object detection."""
    try:
        if 'image' not in request.FILES:
            return Response(
                {'error': 'No image provided', 'objects': []},
                status=status.HTTP_400_BAD_REQUEST
            )

        image_file = request.FILES['image']
        
        # Save the file temporarily to get the path
        import tempfile
        import os
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            # Write uploaded file content to temp file
            for chunk in image_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        try:
            # Perform detection on the temporary file
            result = AIService.detect_objects(temp_path)
            
            # Save detection result to database with the result
            detection = DetectionResult.objects.create(
                image=image_file,
                result=result
            )
            
            return Response(result, status=status.HTTP_200_OK)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        return Response(
            {'error': str(e), 'objects': []},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


#@api_view(['POST'])
#def analyze_sentiment(request):
#    """API endpoint for sentiment analysis."""
#    data = request.data
    
#    if 'text' not in data:
#        return Response(
#            {'error': 'No text provided'},
#            status=status.HTTP_400_BAD_REQUEST
#        )
    
#    text = data['text']
    
    # Perform sentiment analysis
#    result = AIService.analyze_sentiment(text)
    
    # Save to database
#    sentiment_map = {
#        'positive': 'positive',
#        'negative': 'negative',
#        'neutral': 'neutral'
#    }
    
#    sentiment_value = sentiment_map.get(result.get('sentiment', 'neutral'), 'neutral')
    
#    SentimentAnalysis.objects.create(
#        text=text,
#        sentiment=sentiment_value,
#        score=result.get('score', 0.0)
#    )
    
#    return Response(result, status=status.HTTP_200_OK)


@api_view(['POST'])
def chat(request):
    """API endpoint for chatbot."""
    data = request.data
    
    if 'message' not in data:
        return Response(
            {'error': 'No message provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    message = data['message']
    
    # Get AI response
    result = AIService.chat(message)
    response_text = result.get('response', '')
    
    # Save to database
    chat_message = ChatMessage.objects.create(
        message=message,
        response=response_text
    )
    
    return Response({
        'message': message,
        'response': response_text,
        'id': chat_message.id
    }, status=status.HTTP_200_OK)


@api_view(['GET'])
def chat_history(request):
    """Get chat history."""
    chats = ChatMessage.objects.all()[:50]
    data = [
        {
            'id': chat.id,
            'message': chat.message,
            'response': chat.response,
            'created_at': chat.created_at
        }
        for chat in chats
    ]
    return Response(data, status=status.HTTP_200_OK)


from django.shortcuts import render
from django.http import JsonResponse
import json
from .services.sentiment_service import SentimentAnalyzer1

sentiment_model = SentimentAnalyzer1()


#def sentiment_analysis(request):
#    return render(request, "sentiment_analysis.html")


#def sentiment_api(request):
#    if request.method == "POST":
#        try:
#            data = json.loads(request.body)
#            text = data.get("text")

#            result = sentiment_model.analyze(text)

            # Ensure everything is JSON-serializable
#            response_data = {
#                "sentiment": str(result.get("label", "unknown")),
#                "score": float(result.get("score", 0.0)),
#                "explanation": str(result.get("explanation", "Analysis complete"))
#            }

#            return JsonResponse(response_data)

#        except Exception as e:
#            return JsonResponse({"error": str(e)}, status=500)

#    return JsonResponse({"error": "Invalid request"}, status=400)








# core/views.py

# ... (keep your other imports)
from django.views.decorators.csrf import ensure_csrf_cookie

# 1. TEMPLATE VIEW (Returns HTML)
@require_http_methods(["GET"])
@ensure_csrf_cookie # Important for the JS to get the token
def sentiment_page(request):
    """Render the sentiment analysis page."""
    return render(request, 'sentiment_analysis.html')




# 2. API VIEW (Returns JSON)
@api_view(['POST'])
def sentiment_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            text = data.get("text")
            
            result = sentiment_model.analyze(text)

            # If result is just a string, handle it gracefully
            if isinstance(result, str):
                response_data = {
                    "sentiment": result,
                    "score": 0.0,
                    "explanation": "No detailed explanation available."
                }
            else:
                # If it's already a dictionary, use .get()
                response_data = {
                    "sentiment": str(result.get("label", "unknown")),
                    "score": float(result.get("score", 0.0)),
                    "explanation": str(result.get("explanation", "Analysis complete"))
                }
            
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
        


from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# Updated import
from .services.ai_services import ObjectDetector
@api_view(['POST'])
def detect_objects1(request):
    if 'image' not in request.FILES:
        return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        image_file = request.FILES['image']
        # Read image directly as bytes
        image_bytes = image_file.read()
        
        # Call the renamed class and method
        result = ObjectDetector.detect(image_bytes)
        
        # Format the response for your JavaScript
        #return Response({
        #    "objects": [f"{result['name']} ({result['confidence']:.1f}%)"],
        #    "raw_data": result
        #}, status=status.HTTP_200_OK)
        return Response(
            {
                "objects": [result["label"]],
                "confidence": result["confidence"]
            },
            status=status.HTTP_200_OK
        )
                
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    




@api_view(['POST'])
def detect_objects(request):
    # 1. Check if the image exists in the request
    if 'image' not in request.FILES:
        return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        image_file = request.FILES['image']
        image_bytes = image_file.read()
        
        # 2. Get the full analysis from the ObjectDetector
        # This now returns: {"top_predictions": [{"label": "...", "confidence": ...}, ...]}
        result = ObjectDetector.detect(image_bytes)
        
        # 3. Return the entire result dictionary. 
        # Your JavaScript function uses 'data.top_predictions' to build the UI.
        return Response(result, status=status.HTTP_200_OK)
                
    except Exception as e:
        # This catches things like 'KeyError' or 'Model loading' errors
        return Response({'error': f"Detection Error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)    
    
    
    
    
    
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from core.services.ai_services import rag_query


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from core.services.ai_services import rag_query


import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from core.services.ai_services import rag_query


@csrf_exempt
def chat_api(request):

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    user_message = (data.get("message") or "").strip()
    if not user_message:
        return JsonResponse({"error": "No message provided"}, status=400)

    # Conversation history lives in the user's session, not in a process-wide
    # global, so concurrent users do not see each other's context.
    history = request.session.get("chat_history", [])

    try:
        reply = rag_query(user_message, history)
    except Exception as e:
        # Usually Workers AI being slow or unavailable: say so in the user's language.
        language = conversation_language(user_message, history)
        return JsonResponse({"error": f"Chat error: {e}", "response": SERVICE_BUSY_REPLY[language]}, status=503)

    history.append([user_message, reply["answer"]])
    request.session["chat_history"] = history[-settings.RAG_HISTORY_TURNS:]

    ChatMessage.objects.create(message=user_message, response=reply["answer"])

    return JsonResponse({"response": reply["answer"], "sources": reply["sources"]})
