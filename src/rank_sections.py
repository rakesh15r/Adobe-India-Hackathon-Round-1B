from sentence_transformers import util

def rank_sections(sections, task_description, sentence_model):
    task_embedding = sentence_model.encode(task_description, convert_to_tensor=True)

    for section in sections:
        section_embedding = sentence_model.encode(section["section_title"], convert_to_tensor=True)
        similarity = util.cos_sim(section_embedding, task_embedding).item()
        section["relevance_score"] = similarity

    # Sort by highest similarity
    ranked = sorted(sections, key=lambda x: x["relevance_score"], reverse=True)
    
    for i, section in enumerate(ranked):
        section["importance_rank"] = i + 1

    return ranked
