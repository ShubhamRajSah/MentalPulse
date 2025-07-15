from utils.text_cleaner import clean_text
from utils.sentiment_engine import generate_wordcloud, get_sentiment, predict_emotion, explain_emotion
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="MentalPulse", layout="centered")
st.title("🧠 MentalPulse - Daily Mood Analyzer")

# Input
journal = st.text_area("Write your journal entry here:", height=250)
if st.button("Analyze Mood") and journal.strip():
    cleaned_entry=clean_text(journal)
    # Sentiment Analysis
    mood,score= get_sentiment(cleaned_entry)
    clf_mood=predict_emotion(cleaned_entry)

    st.subheader(f"VADER Mood: {mood}")
    st.write(f"VADER Score:`{score:.3f}`")

    st.subheader(f'Classifier mood: {clf_mood}')

    # Explain classifier prediction using SHAP
    explanation = explain_emotion(cleaned_entry)

    # Display explanation
    st.subheader("🔍 Why this mood?")

    import numpy as np  # If not already imported

    for word, impact_val in explanation:        
        icon = "🌟" if impact_val > 0 else "🚫" if impact_val < 0 else "⚖"
        st.markdown(f"{icon} *{word}* → impact score: `{impact_val:.3f}`")
    if all(abs(val)<0.001 for _, val in explanation):
        st.markdown("This journal entry triggered very low emotional signals.")
    # Generate word cloud
    wordcloud = generate_wordcloud(cleaned_entry)
    # Display it
    st.subheader("🌥 Emotional Word Cloud")
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.markdown("Submit some thoughts to unlock your mood insights!")

