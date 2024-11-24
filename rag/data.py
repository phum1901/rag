from pathlib import Path
import PyPDF2

def extract_pdf(row):
    source = row['path'].name 
    with open(row['path'], 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)
        extracted_page = []
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            extracted_page.append({'page_num': page_num + 1, 'text': text, 'source': source})
    return extracted_page