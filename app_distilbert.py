from flask import Flask, request, jsonify, render_template
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import requests

# Load model
model = DistilBertForSequenceClassification.from_pretrained('./distilbert_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('./distilbert_model')

model.eval()

app = Flask(__name__)
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"

def fetch_news(query="latest"):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return [a.get('title','') + " " + a.get('description','') for a in articles if a.get('title')]
    except:
        return []

def predict_text(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()
    label = "Real" if pred == 1 else "Fake"
    return label, confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_text", methods=["POST"])
def predict_text_route():
    data = request.get_json()
    news_text = data.get('text', '')
    if not news_text:
        return jsonify({"error":"No text provided"}), 400
    label, confidence = predict_text(news_text)
    return jsonify({"prediction": label, "confidence": round(confidence*100,2)})

@app.route("/fetch_and_predict", methods=["POST"])
def fetch_and_predict_route():
    data = request.get_json()
    query = data.get('query','latest')
    articles = fetch_news(query)
    results = []
    for article in articles:
        label, confidence = predict_text(article)
        results.append({"text": article, "prediction": label, "confidence": round(confidence*100,2)})
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)

