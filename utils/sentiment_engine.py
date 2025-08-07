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

def explain_emotion(text):
    try:
        vec_sparse = vectorizer.transform([text])
        vec = vec_sparse.toarray()
        feature_names = vectorizer.get_feature_names_out()

        # Dynamically set SHAP evaluation budget
        num_features = vec.shape[1]
        min_evals = 2 * num_features + 1

        # ✅ Use Independent masker for stability
        X_summary = shap.sample(X_dense, 50)
        masker = shap.maskers.Independent(X_summary)
        explainer = shap.explainers.Permutation(classifier.predict_proba, masker)

        # Run SHAP with logging
        shap_values = explainer(vec, max_evals=min_evals)
        print("🔍 SHAP values shape:", shap_values.values.shape)

        predicted_label = classifier.predict(vec)[0]
        if predicted_label not in classifier.classes_:
            return [("Unknown emotion detected.", 0.0)]

        class_index = list(classifier.classes_).index(predicted_label)

        # 🛡 Handle SHAP output shape safely
        values = shap_values.values
        if len(values.shape) == 1:
            word_scores_matrix = values
        elif values.shape[0] == 1 and values.shape[1] > class_index:
            word_scores_matrix = values[0, class_index]
        elif values.shape[0] > 1 and values.shape[1] > class_index:
            word_scores_matrix = values[:, class_index]
        else:
            print("⚠ SHAP output dimension mismatch.")
            return [("Explanation unavailable due to shape mismatch.", 0.0)]

        activated_tokens = set(vectorizer.inverse_transform(vec_sparse)[0])
        top_indices = np.argsort(np.abs(word_scores_matrix))[::-1]
        top_features = []

        for i in top_indices:
            token = feature_names[i]
            score = float(word_scores_matrix[i])
            if token in activated_tokens:
                top_features.append((token, score))
            if len(top_features) >= 10:
                break

        if not top_features:
            return [("Your journal felt emotionally smooth — no strong signals detected.", 0.0)]

        return top_features

    except Exception as e:
        print("💥 SHAP explainability failed:", str(e))
        return [("Explanation unavailable", 0.0)]