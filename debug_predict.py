import sys
import os
import numpy as np
import pickle

# Add project root to path
sys.path.append(os.path.abspath('.'))

from src.model_utils import predict_sentiment, vectorizer, VOCABULARY, MODEL
from src.preprocessing import clean_text

def debug_prediction(text):
    print(f"--- Testing: {text} ---")
    cleaned = clean_text(text)
    print(f"Cleaned Text: '{cleaned}'")
    
    vectorized = vectorizer([cleaned])
    active_indices = np.where(vectorized[0] == 1)[0]
    active_words = [VOCABULARY[i] for i in active_indices]
    
    print(f"Words found in Vocabulary: {active_words}")
    
    if MODEL:
        # Get coefficients for these words
        coeffs = MODEL.coef_[0]
        intercept = MODEL.intercept_[0]
        
        print(f"Intercept (Bias): {intercept:.4f}")
        
        score = intercept
        print("Breakdown:")
        print(f"  Base Intercept: {intercept:.4f}")
        for word in active_words:
            idx = VOCABULARY.index(word)
            w = coeffs[idx]
            score += w
            print(f"  + word '{word}': {w:.4f} (new score: {score:.4f})")
            
        probs = MODEL.predict_proba(vectorized)[0]
        prediction = MODEL.predict(vectorized)[0]
        label = "Negative" if prediction == 1 else "Positive"
        
        print(f"Final Score (logit): {score:.4f}")
        print(f"Prediction: {label} (Label {prediction})")
        print(f"Probabilities: Positive={probs[0]:.4f}, Negative={probs[1]:.4f}")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    # Test the specific input the user mentioned
    debug_prediction("i like it, super")
    # Test a clearly positive one
    debug_prediction("great product")
    # Test a clearly negative one
    debug_prediction("this is worst ever")
    # Test one that works
    debug_prediction("i am happy")
