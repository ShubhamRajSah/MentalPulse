import re
import string
import nltk
from nltk.corpus import stopwords

# Download NLTK resources once
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove stopwords
    tokens = text.split()
    filtered = [word for word in tokens if word not in STOPWORDS]

    return ' '.join(filtered)