from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import pickle
import os

# --- Ensure NLTK resources are available ---
def ensure_nltk_data():
    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet'
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

ensure_nltk_data()

# --- Flask setup ---
app = Flask(__name__)
CORS(app)

# --- File paths ---
MODEL_PATH = 'chatbot_model.pkl'
DATA_PATH = 'college_faq.csv'

# --- NLP preprocessing ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens if token.isalnum() and token not in stop_words
    ]
    return ' '.join(tokens)

# --- Load dataset ---
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        print("⚠️ Dataset not found: college_faq.csv")
        return None

# --- Train or load model ---
def get_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        df = load_data()
        if df is None:
            return None
        X = df['question'].apply(preprocess_text)
        y = df['intent']
        model = make_pipeline(TfidfVectorizer(), SVC(probability=True))
        model.fit(X, y)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
    return model

# --- Initialize model & data ---
df = load_data()
model = get_model()

# --- Default response ---
DEFAULT_RESPONSE = "I'm not sure about that. Try rephrasing or contact info@college.edu."

# --- Routes ---
@app.route('/')
def home():
    return "✅ Flask Chatbot API is running!"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        if not user_input or not model or df is None:
            return jsonify({'response': DEFAULT_RESPONSE})

        processed_input = preprocess_text(user_input)
        intent = model.predict([processed_input])[0]
        if intent in df['intent'].values:
            response = df[df['intent'] == intent]['response'].iloc[0]
        else:
            response = DEFAULT_RESPONSE

        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"⚠️ Server error: {str(e)}"})

# --- Run App ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
