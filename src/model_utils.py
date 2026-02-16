import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'static', 'model', 'model.pickle')
VOCAB_PATH = os.path.join(BASE_DIR, 'static', 'model', 'vocabulary.txt')

def load_vocabulary():
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'r', encoding='utf-8') as file:
            return file.read().splitlines()
    return []

VOCABULARY = load_vocabulary()

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            return pickle.load(file)
    return None

MODEL = load_model()

def vectorizer(cleaned_text_list, vocabulary=VOCABULARY):
    """
    Manually creates a document-term matrix based on the provided vocabulary.
    Same logic as used in the model building notebook.
    """
    vectorized_lst = []

    for sentence in cleaned_text_list:
        sentence_lst = np.zeros(len(vocabulary))
        sentence_words = set(sentence.split())

        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence_words:
                sentence_lst[i] = 1

        vectorized_lst.append(sentence_lst)

    return np.asarray(vectorized_lst, dtype=np.float32)

def predict_sentiment(text):
    """
    Full pipeline for a single prediction.
    """
    from .preprocessing import clean_text
    
    cleaned = clean_text(text)
    vectorized = vectorizer([cleaned])
    
    if MODEL:
        prediction = MODEL.predict(vectorized)
        return "Negative" if prediction[0] == 1 else "Positive"
    return "Model not loaded"
