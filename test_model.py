
from model_utils import load_sentiment_model

classifier = load_sentiment_model()
text = "오늘 기분이 너무 안 좋아요..."
result = classifier(text)

label = result[0]['label']
score = result[0]['score']
print(f"예측 감정: {label}")
print(f"확신도: {score:.2%}")
