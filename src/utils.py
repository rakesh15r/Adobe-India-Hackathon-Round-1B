import numpy as np
import re
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTChar, LTFigure

def overlaps(line_bbox, box_bbox, margin=2):
    lx0, ly0, lx1, ly1 = line_bbox
    bx0, by0, bx1, by1 = box_bbox
    return not (
        lx1 < bx0 - margin or lx0 > bx1 + margin or 
        ly1 < by0 - margin or ly0 > by1 + margin
    )

def extract_title_lines(layout_items):
    page1_items = [item for item in layout_items if item["page"] == 1]
    if not page1_items:
        return []
    max_size = max(item["font_size"] for item in page1_items)
    top_y0_threshold = max(item["y0"] for item in page1_items) * 0.65
    return [
        item["text"].strip()
        for item in page1_items
        if abs(item["font_size"] - max_size) < 0.2 and item["y0"] > top_y0_threshold
    ]

def extract_layout_with_features(pdf_path):
    layout_items = []
    features = []
    box_regions = {}

    for page_num, layout in enumerate(extract_pages(pdf_path)):
        box_regions[page_num] = []
        page_height = layout.bbox[3]
        for el in layout:
            if isinstance(el, (LTTextBox, LTFigure)):
                box_regions[page_num].append((el.x0, el.y0, el.x1, el.y1))

        for element in layout:
            if isinstance(element, LTTextBox):
                for line in element:
                    chars = [char for char in line if isinstance(char, LTChar)]
                    if not chars:
                        continue
                    text = line.get_text().strip()
                    if not text or len(text) < 3:
                        continue
                    font_size = np.mean([c.size for c in chars])
                    is_bold = any("Bold" in c.fontname or "Black" in c.fontname for c in chars)
                    starts_with_number = text[:3].strip()[0].isdigit() if text else False
                    x0, y0, x1, y1 = line.x0, line.y0, line.x1, line.y1
                    line_bbox = (x0, y0, x1, y1)
                    box_overlap = any(overlaps(line_bbox, box) for box in box_regions[page_num])
                    word_count = len(text.split())
                    proximity_to_top = y0 / page_height
                    match = re.match(r'^(\d+(\.\d+)*)(\s+|[\)\.])', text)
                    depth = match.group(1).count('.') + 1 if match else 0
                    layout_items.append({
                        "text": text,
                        "font_size": font_size,
                        "is_bold": is_bold,
                        "starts_with_number": starts_with_number,
                        "y0": y0,
                        "page": page_num + 1,
                        "word_count": word_count,
                        "depth": depth,
                        "proximity_to_top": proximity_to_top,
                        "box_overlap": int(box_overlap)
                    })

    title_lines = extract_title_lines(layout_items)
    filtered_layout = []
    features = []

    for item in layout_items:
        if item["text"].strip() in title_lines and item["page"] == 1:
            continue
        filtered_layout.append(item)
        features.append([
            item["font_size"],
            int(item["is_bold"]),
            int(item["starts_with_number"]),
            item["y0"],
            item["depth"],
            item["word_count"],
            item["proximity_to_top"],
            item["box_overlap"]
        ])
    return title_lines, filtered_layout, np.array(features)

def detect_headings(pdf_path, clf, le, heading_label="heading"):
    title_lines, layout_data, features = extract_layout_with_features(pdf_path)
    title = " ".join(title_lines).strip() if title_lines else ""
    if len(features) == 0:
        return title, []
    preds = clf.predict(features)
    levels = le.inverse_transform(preds)
    # Filter by heading label (if label encoder uses more than two classes)
    outline = []
    for item, lvl in zip(layout_data, levels):
        text = item["text"]
        if len(text) < 4:
            continue
        if item["font_size"] < 10:
            continue
        if not item["is_bold"] and not item["starts_with_number"]:
            continue
        if heading_label is not None and str(lvl) != heading_label:
            continue
        outline.append({
            "level": lvl,
            "text": text,
            "page": item["page"]
        })
    return title, outline

def get_timestamp():
    from datetime import datetime
    return datetime.now().isoformat()
