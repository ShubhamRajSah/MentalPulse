# 🧠 MentalPulse: Emotion-Aware Journal Companion

MentalPulse is a machine learning-powered emotional journal analysis tool. Built for clarity, empathy, and explainability, it doesn’t just predict emotions — it understands them.

---

## 💡 What Does MentalPulse Do?

MentalPulse reads journal-style text entries and accurately identifies the *core emotion* behind them. It explains its prediction with *SHAP-based token attribution*, giving users insight into the emotional signal behind their words.

---

## 🧠 Supported Emotions

- Joy 😊  
- Sadness 😔  
- Numbness 🌫  
- Bittersweet 💔  
- Guilt 😢  
- Disappointment 😞  
- Isolation 🧍  
- Betrayal 💣  
- Hope 🌟  
- Exhaustion 😵  
- Sarcasm 😏  
- Determination 💪

---

## 🔧 How MentalPulse Works

- Vectorizes text using CountVectorizer with 1-3 gram range  
- Trains MultinomialNB classifier on manually curated emotion-labeled data  
- Uses *VADER Sentiment* for mood approximation  
- Deploys *SHAP KernelExplainer* to interpret token-level emotion weights  
- Visualizes emotional contribution with token impact values (Top 10)

---

## 🧪 Model Validation

Over 17 custom diagnostic test sets built using journal-style emotional phrases:

- Emotional contrast pairs (e.g. Sadness vs. Hope, Bittersweet vs. Joy)  
- Trust tests with randomized emotional phrasing  
- New emotion integration: *Determination* tested and validated

✅ Final test passed with 100% emotional resolution across all supported classes.

---

## 🚀 How to Use

```python
from mentalpulse import predict_emotion, explain_emotion, get_sentiment

text = "Falling in love is a great experience, everybody wants to experience. By the way I love coding."

# Predict emotion
emotion = predict_emotion(text)

# Explain prediction with SHAP
explanation = explain_emotion(text)

# Get sentiment from VADER
sentiment, compound_score = get_sentiment(text)

Output example:
# Top Emotional Tokens
🌟 love → impact score: 0.206

🌟 great → impact score: 0.150

⚖ way → impact score: 0.000

Each token shows its weighted contribution to the predicted emotion, offering transparent insight.

---
🧵 Project Philosophy

MentalPulse built by Shubham, a machine learning enthusiast and builder passionate about creating emotionally intelligent AI. This project is not just a classifier — it’s a tribute to empathy in tech, shaped by:

- Rigorous testing  
- Manual contrast tuning  
- A refusal to quit until emotional fluency was achieved

---

📁 File Structure

- text_cleaner.py: filtration by removing stopwords from the text
- sentiment_engine.py: Main prediction engine  
- trainee_data.py: Curated labeled dataset  
- explain_emotion(): SHAP-based token interpreter  
- get_sentiment(): VADER mood reader

---

💙 License & Notes

MentalPulse is an open-source companion for emotional clarity. Adapt it, share it, and refine it with care.

---

🙌 Acknowledgment

To the journalers, builders, and feelers — this project sees you.  
Crafted with grit, tested with compassion.  
It doesn’t just process what you write — it listens.

`