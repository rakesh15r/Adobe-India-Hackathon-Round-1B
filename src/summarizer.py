import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def get_section_text(features, heading_idx, all_heading_idxs):
    start = heading_idx + 1
    next_idxs = [i for i in all_heading_idxs if i > heading_idx]
    end = next_idxs[0] if next_idxs else len(features)
    section_lines = [features[i]['text'] for i in range(start, end)]
    return '\n'.join(section_lines)

def summarize_section(text, job_to_do, st_model, max_sentences=1):
    if not text or not text.strip():
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return ""
    job_vec = st_model.encode([job_to_do])
    sent_vecs = st_model.encode(sentences)
    sims = cosine_similarity(sent_vecs, job_vec).reshape(-1)
    top_sent_idx = np.argsort(sims)[::-1][:max_sentences]
    summary = ' '.join([sentences[i] for i in top_sent_idx])
    return summary.strip()
