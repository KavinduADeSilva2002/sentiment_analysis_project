import re
import string
import pandas as pd
from nltk.stem import PorterStemmer
import os

# Get absolute path to stopwards
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOPWORDS_PATH = os.path.join(BASE_DIR, 'static', 'model', 'corpora', 'stopwords', 'english')

def load_stopwords():
    if os.path.exists(STOPWORDS_PATH):
        with open(STOPWORDS_PATH, 'r') as file:
            return file.read().splitlines()
    return []

STOPWORDS = load_stopwords()
STEMMER = PorterStemmer()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def clean_text(text):
    # lowercase
    text = text.lower()
    
    # remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # remove punctuations
    text = remove_punctuations(text)
    
    # remove digits
    text = re.sub(r'\d+', '', text)
    
    # split and remove stopwords & stemming
    words = text.split()
    cleaned_words = [STEMMER.stem(word) for word in words if word not in STOPWORDS]
    
    return " ".join(cleaned_words)

def preprocess_for_prediction(text_list):
    """
    Takes a list of strings and returns a list of cleaned strings.
    """
    return [clean_text(text) for text in text_list]
