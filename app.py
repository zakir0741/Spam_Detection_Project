from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re

app = Flask(__name__)

model = pickle.load(open('model/spam_model.pkl', 'rb'))
cv = pickle.load(open('model/vectorizer.pkl', 'rb'))

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[^\w\s]', '', text)  
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    cleaned = clean_text(message)  
    vectorized = cv.transform([cleaned]) 
    result = model.predict(vectorized)[0] 
    prediction = "Spam" if result == 1 else "Ham"


    if prediction == "Spam":
        suggestion = "❗ This looks like a scam. Don't click on any links or reply to it."
    else:
        suggestion = "✅ This seems safe. You can proceed if it's from someone you know."

    return render_template('result.html', message=message, prediction=prediction, suggestion=suggestion)

if __name__ == '__main__':
    app.run(debug=True)
