from typing import Optional

from pydantic import BaseModel


class ExtractResult(BaseModel):
    texts: list[str]

class ExtractResponse(BaseModel):
    image: Optional[str] = None
    results: list[str]
    cropped_images: list[str] = None
    