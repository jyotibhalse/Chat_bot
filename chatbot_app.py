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
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load FAQ data
df = pd.read_csv('college_faq.csv')  # Make sure this file is present in the same directory

# Preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words])

# Apply preprocessing to questions
df['processed_question'] = df['question'].apply(preprocess)

# Load or train model
model_path = 'chatbot_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    X = df['processed_question']
    y = df['intent']
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
    model.fit(X, y)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

# Define chatbot API endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    
    if not message.strip():
        return jsonify({'response': 'Please enter a valid question.'})
    
    processed_msg = preprocess(message)
    intent = model.predict([processed_msg])[0]
    
    responses = df[df['intent'] == intent]['response'].tolist()
    response = random.choice(responses) if responses else "Sorry, I don't have an answer for that."
    
    return jsonify({'response': response})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5050)
