from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Specify the local model path relative to your Docker image/app root
LOCAL_MODEL_PATH = "models/paraphrase-MiniLM-L6-v2"

def load_sentence_transformer(model_path=LOCAL_MODEL_PATH):
    # Explicitly load from local directory, prevents internet access!
    return SentenceTransformer(model_path)

def rank_by_relevance(title_items, job_to_do, st_model):
    section_titles = [item['section_title'] for item in title_items]
    if not section_titles:
        return []
    # Do all encoding in a try/except if you want (for robust offline error reporting)
    title_embeddings = st_model.encode(section_titles)
    task_embedding = st_model.encode([job_to_do])

    similarities = cosine_similarity(title_embeddings, task_embedding).reshape(-1)

    # Attach scores
    for item, sim in zip(title_items, similarities):
        item['relevance_score'] = float(sim)

    # Sort descending by score
    return sorted(title_items, key=lambda x: -x['relevance_score'])
