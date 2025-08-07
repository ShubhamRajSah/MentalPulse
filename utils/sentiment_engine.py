import pandas as pd
import numpy as np
from wordcloud import WordCloud
import shap
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib
matplotlib.use("Agg")  # Headless backend for Streamlit

from .trainee_data import df

# VADER setup
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        mood = "🙂 Positive"
    elif compound <= -0.05:
        mood = "🙁 Negative"
    else:
        mood = "😐 Neutral"
    return mood, compound

def generate_wordcloud(text):
    wc = WordCloud(width=600, height=400, background_color='white').generate(text)
    return wc

# Vectorizer
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=3000, min_df=1)
X = vectorizer.fit_transform(df['text'])
X_dense = X.toarray()
y = df['label']

# Classifier
classifier = MultinomialNB()
classifier.fit(X, y)

def predict_emotion(text):
    vec = vectorizer.transform([text])
    return classifier.predict(vec)[0]

def model_predict(X):
    return classifier.predict_proba(X)

def explain_emotion(text, classifier, vectorizer, explainer, class_names):

    # Vectorize input
    vec_sparse = vectorizer.transform([text])
    vec_dense = vec_sparse.toarray()

    # Predict emotion class
    predicted_label = classifier.predict(vec_dense)[0]
    predicted_class_index = list(classifier.classes_).index(predicted_label)

    # Get SHAP values
    values = explainer(vec_dense)
    
    # Debugging outputs
    print("🧠 Input text:", text)
    print("📦 Predicted label:", predicted_label)
    print("📊 SHAP values shape:", values.shape)

    # Check if SHAP values are valid
    if len(values.shape) != 3 or predicted_class_index >= values.shape[1]:
        print("⚠ SHAP output shape mismatch or invalid class index.")
        return [("Could not interpret emotional signals due to model mismatch.", 0.0)]

    # Get SHAP scores for predicted class
    word_scores_matrix = values[0][predicted_class_index]
    feature_names = vectorizer.get_feature_names_out()

    # Get activated tokens
    activated_tokens = set(vectorizer.inverse_transform(vec_sparse)[0])
    print("🎯 Activated tokens:", activated_tokens)

    if not activated_tokens:
        print("⚠ No recognizable tokens found in vectorizer.")
        return [("No recognizable emotional tokens found in your journal.", 0.0)]

    # Pair tokens with SHAP scores
    token_scores = [(token, word_scores_matrix[feature_names.tolist().index(token)])
                    for token in activated_tokens if token in feature_names]

    # Sort by score
    token_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    # Debug top scores
    print("📈 Top SHAP scores")