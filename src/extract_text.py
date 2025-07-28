from PyPDF2 import PdfReader
import os

def extract_pdf_text(filepath):
    reader = PdfReader(filepath)
    text_by_page = {}
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text()
        except Exception:
            txt = ""
        text_by_page[i+1] = txt or ""
    return text_by_page

def extract_from_folder(folder_path, doc_filenames):
    '''Returns dict: {filename: {page_number: text}}'''
    doc_texts = {}
    for fname in doc_filenames:
        path = os.path.join(folder_path, fname)
        doc_texts[fname] = extract_pdf_text(path)
    return doc_texts
