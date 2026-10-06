import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    """
    Sentiment analyzer using nlptown multilingual BERT model.
    """

    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def analyze(self, text: str):

        tokens = self.tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model(tokens)

        score = int(torch.argmax(outputs.logits))

        sentiment = "Negative" if score <= 2 else "Positive"

        return {
            "text": text,
            "score": score,
            "sentiment": sentiment
        }
    


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalyzer1:
    """
    Returns only sentiment label:
        Negative or Positive
    """

    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def analyze(self, text: str) -> str:
        tokens = self.tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model(tokens)

        score = int(torch.argmax(outputs.logits))

        # Map score to sentiment
        return "Negative" if score <= 2 else "Positive"