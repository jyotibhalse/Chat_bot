from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import pickle
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "chatbot_model.pkl"
DATA_PATH = "college_faq.csv"

# NLP preprocessing (no nltk.download needed)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # simple regex tokenizer (no punkt)
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=["question", "intent", "response"])
        return df
    except FileNotFoundError:
        print("Dataset not found. Please create college_faq.csv")
        return None

def get_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        df = load_data()
        if df is None:
            return None
        X = df["question"].apply(preprocess_text)
        y = df["intent"]
        model = make_pipeline(TfidfVectorizer(), SVC(probability=True))
        model.fit(X, y)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
    return model

model = get_model()
df = load_data()

DEFAULT_RESPONSE = "I'm not sure about that. Try rephrasing or contact info@college.edu."

@app.route("/")
def home():
    return "Backend is running on Render!"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message")
        if not user_input or not model or df is None:
            return jsonify({"response": DEFAULT_RESPONSE})

        processed_input = preprocess_text(user_input)
        intent = model.predict([processed_input])[0]
        response = (
            df[df["intent"] == intent]["response"].iloc[0]
            if intent in df["intent"].values
            else DEFAULT_RESPONSE
        )
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
