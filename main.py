import pillow_avif
import base64
import os
import sys
from typing import Literal

from src.crop import crop
from src.extract import extract_from_cropped_images
from src.model import ExtractResponse

yolo_path = os.path.join(os.path.dirname(__file__), 'yolov7') 
sys.path.append(yolo_path)

from src.detection import run_inference
from src.logger import setup_logger
from fastapi import FastAPI, File, HTTPException, UploadFile, status
app = FastAPI()
logger = setup_logger()

images = []


@app.post("/detect")
async def detect_objects(
    image: UploadFile = File(...),
    option: Literal["easy_ocr", "paddle"] = "paddle") -> ExtractResponse:

    """
    Upload JPG image and detect objects using YOLOv7 and easy OCR or paddle for text extraction.
    """

    logger.info("Request received: filename=%s, category=%s",
                image.filename, option)
    
    if image.content_type not in {"image/jpeg", "image/jpg"}:
        logger.warning("Rejected file due to invalid MIME type.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG/JPEG images are allowed."
        )

    if not image.filename.lower().endswith((".jpg", ".jpeg")):
        logger.warning("Rejected file due to invalid extension.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have .jpg or .jpeg extension."
        )

    contents = await image.read()

    try:
        labels, img_bytes, coordinates, overlay_img_bytes = run_inference(contents)
    except ValueError as exc:
        logger.error("Inference failed: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc
    
    logger.info("Inference successful. Detected labels and bounding box: %s, %s", labels, coordinates if coordinates else "None")

    for coordinate in coordinates:
        logger.info("Bounding box coordinates: %s", coordinate)
        cropped_image = crop(img_bytes, coordinate)
        images.append(cropped_image)

    logger.info("Cropped images created successfully. Total cropped images: %d", len(images))

    try:
        logger.info(f"Received file '{image.filename}' for text extraction.")
        extracted_texts = extract_from_cropped_images(images, option)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the image."
        )
    
    logger.info(f"Text extraction successful. Extracted texts: {extracted_texts}")

    base64_encoded_overlay = base64.b64encode(overlay_img_bytes).decode('utf-8')
    
    return ExtractResponse(
        image=base64_encoded_overlay,
        results=extracted_texts,
        cropped_images=[ base64.b64encode(cropped).decode('utf-8') for cropped in images ]
    )
