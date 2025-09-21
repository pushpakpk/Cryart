import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Ensure required resources are available ---
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# --- Initialize ---
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text: str):
    """
    Preprocess the input text:
    1. Lowercase
    2. Tokenize
    3. Remove stopwords & non-alphabetic tokens
    4. Stemming
    """
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words
    ]
    return filtered_tokens
