import joblib
import os

def load_heading_extractor(model_path="model-1.pkl"):
    """
    Load the heading extraction model and label encoder.
    The model file should contain a tuple: (classifier, label_encoder).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")
    return joblib.load(model_path)
