import os
from torch import nn
from models.experimental import attempt_load
from models.yolo import IDetect, Model
from src.logger import setup_logger
import io
import sys
from typing import List
import torch
from PIL import Image
import cv2
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.plots import plot_one_box
from utils.torch_utils import select_device

os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

logger = setup_logger()

sys.path.append("./yolov7")  

torch.serialization.add_safe_globals([
    Model, 
    IDetect, 
    nn.modules.container.Sequential, 
    nn.modules.container.ModuleList,
    nn.modules.conv.Conv2d,
    nn.modules.batchnorm.BatchNorm2d,
    nn.modules.activation.SiLU
])

DEVICE = select_device("cpu")
WEIGHTS_PATH = "./yolov7/best.pt"

logger.info("Loading YOLOv7 model from local weights...")

model = attempt_load(WEIGHTS_PATH, map_location=DEVICE)
model.eval()

logger.info("YOLOv7 model loaded successfully.")

CLASS_NAMES = model.names

def run_inference(image_bytes: bytes) -> List[str]:
    """
    This function takes the image bytes, converts back to the image, resizes it and runs inference on YOLOv7 and returns the predicted labels.

    Parameters
    ----------
    Param: bytes
        a single image stream of bytes

    Returns
    -------
    List[str]
        Returns the predicted labels from YOLOv7

    Raise
    -----
    ValueError
        If the image file is invaid
    """

    logger.info(f"Received bytes size: {len(image_bytes)}")
    logger.info(f"First bytes: {image_bytes[:20]}")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning(f"Invalid image received for inference: {type(exc).__name__} - {exc}")
        raise ValueError(f"Invalid image file: {exc}") from exc

    image = image.resize((640, 640))
    image0 = np.array(image)
    image_raw = image0.copy()

    img_tensor = torch.from_numpy(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
         .float()
         .reshape(640, 640, 3)
         .numpy())
    ).permute(2, 0, 1) / 255.0

    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)[0]

    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    labels: List[str] = []
    coordinates = []

    for det in pred:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                print(f"Detected object with confidence {conf:.2f} at coordinates ({x1}, {y1}), ({x2}, {y2})")
                label = CLASS_NAMES[int(cls)]
                labels.append(label)
                plot_one_box(xyxy, image0, label=label)
                coordinates.append((x1, y1, x2, y2))

    logger.info("coordinates: %s", coordinates)
    logger.info("bounding box coordinates: %s", coordinates[0] if coordinates else "None")

    def encode(img):
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return buffer.tobytes()
    
    img_bytes = encode(image_raw)
    overlay_img_bytes = encode(image0)

    logger.info("Inference completed. Detected %d objects.", len(labels))

    return labels, img_bytes, coordinates, overlay_img_bytes


# def filter_by_category(labels: List[str], category: str) -> List[int]:
#     """
#     This function takes the predicted labels and the input category and filters the labels by that category.

#     Parameters
#     ----------
#     Param1: List[str]
#         List of output categories
#     Param2: String
#         The category to filter from the List

#     Returns
#     -------
#     List[str]
#         Returns the filter List of Strings

#     Example
#     -------
#     >>> filter_by_category(["person"], "cat")
#     returns []

#     >>> filter_by_category(["person"], "person")
#     returns ["person"]
#     """

#     filtered = []

#     if category in CATEGORY_MAP:
#         labels = [
#             label for label in labels if label in CATEGORY_MAP[category]
#         ]
#     else:
#         labels = [
#             label for label in labels if label == category
#         ]

#     logger.info(
#         "Filtering done. Category='%s', matched=%d",
#         category,
#         len(labels)
#     )

#     filtered.append(len(labels))

#     return filtered
