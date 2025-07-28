import os
import json
from src.model_loader import load_heading_extractor   # Not needed unless you want to filter classifier
from src.utils import extract_layout_with_features, get_timestamp
from src.relevance import load_sentence_transformer, rank_by_relevance
from src.summarizer import summarize_section

def main(input_json_path):
    # 1. Load input
    with open(input_json_path, "r") as f:
        input_data = json.load(f)
    docs = input_data["documents"]
    persona = input_data["persona"]["role"]
    job_to_do = input_data["job_to_be_done"]["task"]
    doc_filenames = [d["filename"] for d in docs]

    # 2. Load sentence transformer model
    st_model = load_sentence_transformer()

    all_sections = []
    filtered_layout_by_doc = {}

    for fname in doc_filenames:
        pdf_path = os.path.join("data/input_documents", fname)
        title_lines, filtered_layout, feature_array = extract_layout_with_features(pdf_path)
        filtered_layout_by_doc[fname] = filtered_layout
        if not filtered_layout:
            continue

        # Instead of classifier filtering, use ALL nontrivial lines
        for idx, item in enumerate(filtered_layout):
            text = item['text'].strip()
            if not text or len(text) < 4:
                continue
            all_sections.append({
                'document': fname,
                'section_title': text,
                'page_number': item['page'],
                'feature_idx': idx
            })

    # 3. Rank all candidate lines/sections by semantic similarity to the job
    ranked_sections = rank_by_relevance(all_sections, job_to_do, st_model)
    for i, sec in enumerate(ranked_sections):
        sec['importance_rank'] = i + 1

    # 4. Assemble output's extracted_sections (top 5)
    extracted_sections = [{
        'document': s['document'],
        'section_title': s['section_title'],
        'importance_rank': s['importance_rank'],
        'page_number': s['page_number']
    } for s in ranked_sections[:5]]

    # 5. Assemble output's subsection_analysis (top 5)
    subsection_analysis = []
    for s in ranked_sections[:5]:
        doc = s['document']
        idx = s['feature_idx']
        filtered_layout = filtered_layout_by_doc[doc]
        # Section text: all lines after this index until the next higher candidate, or to document end
        # Find following candidates in same doc
        doc_ranks = [item for item in ranked_sections if item['document'] == doc]
        doc_ranks_sorted = sorted([item['feature_idx'] for item in doc_ranks])
        next_idxs = [i for i in doc_ranks_sorted if i > idx]
        next_idx = next_idxs[0] if next_idxs else len(filtered_layout)
        section_text = '\n'.join(filtered_layout[i]['text'] for i in range(idx + 1, next_idx))
        refined = summarize_section(section_text, job_to_do, st_model, max_sentences=2)
        subsection_analysis.append({
            'document': doc,
            'refined_text': refined,
            'page_number': s['page_number']
        })

    metadata = {
        'input_documents': doc_filenames,
        'persona': persona,
        'job_to_be_done': job_to_do,
        'processing_timestamp': get_timestamp()
    }
    output = {
        'metadata': metadata,
        'extracted_sections': extracted_sections,
        'subsection_analysis': subsection_analysis
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/challenge1b_output.json", "w", encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main("data/sample_input.json")
