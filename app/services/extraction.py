from pdf2image import convert_from_bytes
from PIL import Image
import io
import fitz
import re
import unicodedata
import os
import sys
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def clean_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[•▪◦·¢®©@&%#_=<>^~`\*\|\[\]\{\}]", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\b[eEoO]\b", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = text.strip()
    return text

def extract_text_from_cv(file_bytes: bytes, filename: str) -> str:
    text = ""

    if filename.lower().endswith(".pdf"):
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        for page in pdf:
            page_text = page.get_text("text")
            if page_text.strip():
                text += page_text
        pdf.close()

        if not text.strip():
            try:
                images = convert_from_bytes(file_bytes)
                import pytesseract
                for image in images:
                    text += pytesseract.image_to_string(image)
            except Exception as e:
                print(f"OCR fallback failed for {filename}: {e}")

    elif filename.lower().endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"DOCX extraction failed for {filename}: {e}")

    elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            import pytesseract
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)
        except Exception as e:
            print(f"Image OCR failed for {filename}: {e}")
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    
    text = clean_extracted_text(text)
    return text.strip()
