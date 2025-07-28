def generate_features(text_by_page):
    # For each line on each page, yield page_num, line_text
    features = []
    for page_num, page_text in text_by_page.items():
        if page_text:
            for line in page_text.split('\n'):
                features.append({'page': page_num, 'text': line})
    return features
