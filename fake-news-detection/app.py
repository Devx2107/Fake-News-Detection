from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from newspaper import Article

app = Flask(__name__)
CORS(app)

# -------------------------
# Load AI models
# -------------------------

classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

category_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Explanation model (working version)
explainer_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
explainer_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# -------------------------
# History storage
# -------------------------

history = []

# -------------------------
# Serve frontend
# -------------------------

@app.route("/")
def home():
    return send_file("index.html")

# -------------------------
# Prediction endpoint
# -------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    text = data.get("text")
    url = data.get("url")

    # -------- URL support --------

    if url:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

    if not text:
        return jsonify({"error": "No text or URL provided"}), 400

    # limit input length
    text = text[:500]

    # -------- Fake / Real classification --------

    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    if label.lower() == "fake":
        prediction = "Fake News"
    else:
        prediction = "Real News"

    # -------- Category detection --------

    categories = [
        "Health misinformation",
        "Political misinformation",
        "Conspiracy theory",
        "Financial scam",
        "Legitimate news"
    ]

    category_result = category_classifier(
        text,
        candidate_labels=categories
    )

    category = category_result["labels"][0]

    # -------- Override prediction if category signals misinformation --------

    if category in [
        "Health misinformation",
        "Political misinformation",
        "Conspiracy theory",
        "Financial scam"
    ]:
        prediction = "Fake News"

    # -------- Credibility score --------

    credibility = score * 100

    if category == "Conspiracy theory":
        credibility *= 0.2
    elif category == "Health misinformation":
        credibility *= 0.3
    elif category == "Financial scam":
        credibility *= 0.3
    elif category == "Political misinformation":
        credibility *= 0.4

    credibility = round(credibility, 2)

    # -------- Simple fact result --------

    if prediction == "Fake News":
        fact_result = "False"
    else:
        fact_result = "True"

    # -------- Explanation generation --------

    prompt = f"Explain in one short sentence why this news might be fake or real: {text}"

    inputs = explainer_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = explainer_model.generate(**inputs, max_new_tokens=80, do_sample=False)
    explanation = explainer_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if len(explanation) < 5:
        explanation = "The claim lacks reliable evidence or contradicts established scientific knowledge."

    if "." in explanation:
        explanation = explanation.split(".")[0] + "."

    # -------- Store history --------

    history.append({
        "text": text[:120],
        "prediction": prediction,
        "confidence": round(score * 100, 2),
        "category": category
    })

    if len(history) > 20:
        history.pop(0)

    # -------- Response --------

    return jsonify({
        "prediction": prediction,
        "confidence": round(score * 100, 2),
        "credibility": credibility,
        "category": category,
        "fact_check": fact_result,
        "explanation": explanation
    })


# -------------------------
# History endpoint
# -------------------------

@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(history)

# -------------------------
# Run server
# -------------------------

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
