import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text']
y = df['label']

cv = CountVectorizer()
X_vec = cv.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

import os
os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/spam_model.pkl', 'wb'))
pickle.dump(cv, open('model/vectorizer.pkl', 'wb'))

print("✅ Model and vectorizer saved successfully.")
