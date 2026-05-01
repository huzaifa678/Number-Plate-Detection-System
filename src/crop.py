import cv2
import numpy as np

def crop(image: bytes, coordinates: tuple, margin_percent: float = 0.1) -> bytes:
    x1, y1, x2, y2 = map(int, coordinates)

    np_img = np.frombuffer(image, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if (x1, y1) >= (150, 270):
        bytes = crop_with_margin(image, coordinates, margin_percent)
        return bytes
    else:
        cropped_image = img[y1:y2, x1:x2]
        _, buffer = cv2.imencode(".jpg", cropped_image)
        return buffer.tobytes()

def crop_with_margin(image: bytes, coordinates: tuple, margin_percent: float = 0.1) -> bytes:
    x1, y1, x2, y2 = map(int, coordinates)

    np_img = np.frombuffer(image, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    w, h = x2 - x1, y2 - y1
    
    nx1 = max(0, x1 - int(w * margin_percent))
    ny1 = max(0, y1 - int(h * margin_percent))
    nx2 = min(img.shape[1], x2 + int(w * margin_percent))
    ny2 = min(img.shape[0], y2 + int(h * margin_percent))

    cropped_image = img[ny1:ny2, nx1:nx2]
    _, buffer = cv2.imencode(".jpg", cropped_image)
    return buffer.tobytes()
