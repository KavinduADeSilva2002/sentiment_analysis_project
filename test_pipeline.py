import sys
import os

# Test imports and basic functionality
sys.path.append(os.path.abspath('.'))
from src.model_utils import predict_sentiment

test_text = "I absolutely love this! It's fantastic."
result = predict_sentiment(test_text)
print(f"Test Result: {result}")
