import io
from typing import Literal
from PIL import Image
import numpy as np
from easyocr import easyocr
from paddleocr import PaddleOCR
from src.logger import setup_logger
from src.model import ExtractResult

logger = setup_logger()
reader = easyocr.Reader(['en'], gpu=False)
logger.info("Downloading PaddleOCR files...")
paddle_ocr = PaddleOCR(lang='en') 

def extract_text_from_image(image_bytes: bytes, option: Literal["easy_ocr", "paddle"]) -> ExtractResult:
    """
    This function takes the image bytes and the OCR option, converts back to the image, and runs OCR to extract text from the image.

    Parameters
    ----------
    Param: bytes
        a single image stream of bytes
    Param: option
        a string that specifies which OCR method to use ("easy_ocr" or "paddle)

    Returns
    -------
    List[str]
        Returns the extracted text lines

    Raise
    -----
    ValueError
        If the image file is invaid
    """
    
    logger.info("Starting text extraction from image.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info("Image loaded successfully for processing.")
    except Exception as exc:
        logger.warning("Invalid image received for inference.")
        raise ValueError("Invalid image file.") from exc
    
    image_np = np.array(image)

    if option == "paddle":
        logger.info("Using PaddleOCR for text extraction.")

        results_raw = paddle_ocr.ocr(image_np)

        def extract_paddle_text(results_raw):
            return [line[1][0] for line in results_raw[0] or []]

        results = extract_paddle_text(results_raw)

        logger.info("PaddleOCR model loaded successfully.")
        logger.info(f"Extracted {len(results)} text lines using PaddleOCR.")
        logger.info(f"Extracted texts from PADDLE OCR: {results}")
    else:
        logger.info("Using EasyOCR for text extraction.")
        results = reader.readtext(image_np)
        results = [text for _, text, _ in results]
        logger.info(f"Extracted {len(results)} text lines using EasyOCR.")

    return ExtractResult(texts=results)

def extract_from_cropped_images(cropped_images: list, option: Literal["easy_ocr", "paddle"]) -> list:
    extracted_texts = []
    logger.info(f"Starting text extraction from {len(cropped_images)} cropped images using {option}.")
    for img in cropped_images:
        logger.info("Extracting text from cropped image.")
        result = extract_text_from_image(img, option)
        extracted_texts.extend(result.texts)
        logger.info(f"Extracted texts from cropped image: {result.texts}")
        logger.info(f"Total extracted texts so far: {extracted_texts}")
    return extracted_texts
