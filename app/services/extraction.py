from pdf2image import convert_from_bytes
from PIL import Image, ImageOps
import io
import fitz
import re
import unicodedata
import os
from typing import Tuple
import pytesseract

pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng+fra")
OCR_CONF_THRESHOLD = float(os.getenv("OCR_MIN_CONFIDENCE", "35.0"))


def clean_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[•▪◦·¢®©@&%#_=<>^~`\*\|\[\]\{\}]", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\b[eEoO]\b", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = text.strip()
    return text


def _run_ocr(image: Image.Image, filename: str) -> Tuple[str, float]:
    gray_image = ImageOps.grayscale(image)
    data = pytesseract.image_to_data(
        gray_image,
        lang=OCR_LANGUAGES,
        output_type=pytesseract.Output.DICT,
    )

    confidences = [float(conf) for conf in data.get("conf", []) if conf not in ("-1", "-1.0")]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    text = " ".join(
        data["text"][i] for i, conf in enumerate(data["conf"]) if conf not in ("-1", "-1.0")
    )

    if avg_conf < OCR_CONF_THRESHOLD:
        # Retry with adjusted parameters
        try:
            retry_text = pytesseract.image_to_string(
                gray_image,
                lang=OCR_LANGUAGES,
                config="--oem 3 --psm 6",
            )
            retry_data = pytesseract.image_to_data(
                gray_image,
                lang=OCR_LANGUAGES,
                config="--oem 3 --psm 6",
                output_type=pytesseract.Output.DICT,
            )
            retry_confidences = [
                float(conf)
                for conf in retry_data.get("conf", [])
                if conf not in ("-1", "-1.0")
            ]
            retry_avg = (
                sum(retry_confidences) / len(retry_confidences) if retry_confidences else avg_conf
            )
            if retry_avg > avg_conf:
                text = retry_text
                avg_conf = retry_avg
        except Exception as retry_error:
            print(f"OCR retry failed for {filename}: {retry_error}")

    if avg_conf < OCR_CONF_THRESHOLD:
        print(f"[OCR Warning] Low confidence ({avg_conf:.2f}) for {filename}")

    return text, avg_conf


def extract_text_from_cv(file_bytes: bytes, filename: str) -> str:
    text = ""

    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            for page in pdf:
                page_text = page.get_text("text")
                if page_text.strip():
                    text += page_text
        finally:
            pdf.close()

        if not text.strip():
            try:
                images = convert_from_bytes(file_bytes)
                for image in images:
                    ocr_text, _ = _run_ocr(image, filename)
                    text += "\n" + ocr_text
            except Exception as e:
                print(f"OCR fallback failed for {filename}: {e}")

    elif filename_lower.endswith(".docx"):
        try:
            import docx

            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"DOCX extraction failed for {filename}: {e}")

    elif filename_lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        try:
            image = Image.open(io.BytesIO(file_bytes))
            ocr_text, _ = _run_ocr(image, filename)
            text = ocr_text
        except Exception as e:
            print(f"Image OCR failed for {filename}: {e}")
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    text = clean_extracted_text(text)
    return text.strip()
