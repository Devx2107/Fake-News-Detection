import streamlit as st
import pickle
import os
import re

# -------------------------
# Page config
# -------------------------

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🧠",
    layout="centered"
)

# -------------------------
# Custom CSS to match TruthLens design
# -------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(120deg, #0f172a, #1e3a8a, #6366f1);
    background-size: 400% 400%;
    animation: gradientMove 20s ease infinite;
}

@keyframes gradientMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    padding: 35px;
    margin: 10px 0 20px 0;
}

.title-text {
    text-align: center;
    color: white;
    font-size: 46px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle-text {
    text-align: center;
    color: #cbd5f5;
    font-size: 14px;
    margin-bottom: 30px;
}

.result-fake {
    color: #ef4444;
    font-size: 26px;
    font-weight: 700;
}

.result-real {
    color: #22c55e;
    font-size: 26px;
    font-weight: 700;
}

.detail-box {
    background: rgba(0,0,0,0.4);
    border-radius: 12px;
    padding: 20px 25px;
    color: white;
    margin-top: 15px;
    line-height: 1.8;
}

.detail-box b {
    color: #a5b4fc;
}

.history-item {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    color: white;
    font-size: 13px;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    font-size: 15px !important;
}

.stButton > button {
    width: 100%;
    background: #6366f1;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #4f46e5;
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

label, .stTextArea label {
    color: white !important;
    font-size: 15px !important;
}

.meter-container {
    background: #1e293b;
    border-radius: 10px;
    height: 18px;
    overflow: hidden;
    margin: 12px 0;
}

.meter-fill-fake {
    height: 100%;
    border-radius: 10px;
    background: #ef4444;
    transition: width 1.2s ease;
}

.meter-fill-warn {
    height: 100%;
    border-radius: 10px;
    background: #f59e0b;
}

.meter-fill-real {
    height: 100%;
    border-radius: 10px;
    background: #22c55e;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load models
# -------------------------

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "fake_news_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_models()

# -------------------------
# Helpers
# -------------------------

MISINFORMATION_KEYWORDS = {
    "conspiracy":  ["illuminati", "deep state", "new world order", "chemtrail", "microchip",
                    "mind control", "secret society", "false flag"],
    "health":      ["miracle cure", "doctors don't want", "big pharma hiding", "bleach cure",
                    "vaccine causes", "cures cancer", "detox", "natural remedy eliminates"],
    "political":   ["rigged election", "stolen vote", "deep state plot", "government cover"],
    "financial":   ["guaranteed returns", "get rich quick", "investment secret", "crypto millionaire"],
}

CATEGORY_DISPLAY = {
    "conspiracy":   "Conspiracy theory",
    "health":       "Health misinformation",
    "political":    "Political misinformation",
    "financial":    "Financial scam",
    "general news": "Legitimate news",
}

CREDIBILITY_MULTIPLIERS = {
    "conspiracy":   0.15,
    "health":       0.25,
    "political":    0.35,
    "financial":    0.25,
    "general news": 1.0,
}

EXPLANATIONS = {
    "conspiracy":   "The article contains language commonly associated with conspiracy theories and lacks credible sourcing.",
    "health":       "The article makes health claims that contradict established medical consensus.",
    "political":    "The article uses politically charged language typical of misinformation campaigns.",
    "financial":    "The article promotes financial claims with hallmarks of scam or fraud content.",
    "general news": None,
}

def detect_category(text):
    text_lower = text.lower()
    for category, keywords in MISINFORMATION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return category
    return "general news"

def predict(text):
    text = text[:500]
    vec   = vectorizer.transform([text])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    if label == 0:
        prediction = "Fake News"
        confidence = round(proba[0] * 100, 2)
    else:
        prediction = "Real News"
        confidence = round(proba[1] * 100, 2)

    category_key     = detect_category(text)
    category_display = CATEGORY_DISPLAY[category_key]

    if category_key != "general news":
        prediction = "Fake News"
        confidence = min(round((1 - proba[1]) * 100 + 15, 2), 97.0)

    base_score  = confidence
    credibility = round(base_score * CREDIBILITY_MULTIPLIERS[category_key], 2)
    credibility = max(1.0, min(credibility, 99.0))

    fact_check  = "False" if prediction == "Fake News" else "Likely True"

    static_exp  = EXPLANATIONS.get(category_key)
    if static_exp and prediction == "Fake News":
        explanation = static_exp
    elif prediction == "Fake News":
        explanation = "The text exhibits patterns commonly found in fabricated or misleading news articles."
    else:
        explanation = "The text aligns with patterns found in credible, factual reporting."

    return {
        "prediction":  prediction,
        "confidence":  confidence,
        "credibility": credibility,
        "category":    category_display,
        "fact_check":  fact_check,
        "explanation": explanation,
    }

# -------------------------
# Session state for history
# -------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# UI
# -------------------------

st.markdown('<div class="title-text">🧠 TruthLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Detect misinformation and verify credibility using AI</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

news_text = st.text_area("Paste news text", height=180, placeholder="Paste a news article or headline here...")

analyze = st.button("🔍 Analyze News")

if analyze:
    if not news_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("AI scanning article..."):
            result = predict(news_text)

        st.session_state.history.insert(0, {
            "text":       news_text[:120],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "category":   result["category"],
        })
        if len(st.session_state.history) > 20:
            st.session_state.history.pop()

        # Result label
        if result["prediction"] == "Fake News":
            st.markdown('<div class="result-fake">🚨 Fake News</div>', unsafe_allow_html=True)
            bar_class = "meter-fill-fake"
        else:
            st.markdown('<div class="result-real">✅ Real News</div>', unsafe_allow_html=True)
            bar_class = "meter-fill-real" if result["credibility"] >= 70 else "meter-fill-warn"

        # Trust bar
        st.markdown(f"""
        <div class="meter-container">
            <div class="{bar_class}" style="width:{result['credibility']}%"></div>
        </div>
        """, unsafe_allow_html=True)

        # Detail box
        st.markdown(f"""
        <div class="detail-box">
            <b>Confidence:</b> {result['confidence']}%<br>
            <b>Trust Score:</b> {result['credibility']}/100<br>
            <b>Category:</b> {result['category']}<br>
            <b>Fact Check:</b> {result['fact_check']}<br><br>
            <b>Explanation:</b> {result['explanation']}
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# History
# -------------------------

if st.session_state.history:
    st.markdown("### 🕒 Recent Checks")
    for item in st.session_state.history:
        icon = "🚨" if item["prediction"] == "Fake News" else "✅"
        st.markdown(f"""
        <div class="history-item">
            <b>{icon} {item['prediction']}</b> ({item['confidence']}%) — {item['category']}<br>
            <span style="opacity:0.7">{item['text']}...</span>
        </div>
        """, unsafe_allow_html=True)
