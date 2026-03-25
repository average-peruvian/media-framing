import torch
from transformers import BertForSequenceClassification, BertTokenizer

from ..distrib import BaseWorker

class Sentimentalist(BaseWorker):
    def __init__(self, 
            hf_model = "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis",
            threshold = 0.5,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.model = BertForSequenceClassification.from_pretrained(hf_model)
        self.tokenizer = BertTokenizer.from_pretrained(hf_model)
        self.threshold = threshold

    def process_row(self, row):
        inputs = self.tokenizer(row['body'], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
        predicted_class = torch.argmax(logits, dim=1).item()
        if probabilities[predicted_class] <= threshold and predicted_class == 1:
            predicted_class = 0

        return {
            "id":           row['id'],
            "class":        predicted_class,
            "confidence":   probabilities[predicted_class]
        }

    def on_error(self, row, exc: Exception) -> dict | None:
        return {
            "id":           row['id'],
            "class":        None,
            "confidence":   None
        }