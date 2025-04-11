
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def load_sentiment_model():
    model_name = "beomi/kcbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier
