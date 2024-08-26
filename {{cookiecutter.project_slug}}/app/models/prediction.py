import io
from typing import Any, Dict, List
from urllib.request import urlopen
from fastapi import HTTPException

from PIL import Image
from pydantic import BaseModel


class DetectionResult(BaseModel):
    bbox: List[int]
    class_name: str
    detection_score: float


class ClassificationResult(BaseModel):
    class_name: int
    classification_score: float


class MachineLearningResponse(BaseModel):
    status_code: int
    # a dictionary derived from ClassificationResult
    content: Dict[str, Any]
    # result: Dict[
    #     DetectionResult
    # ]  # Change to Dict[ClassificationResult] if classification task or Dict[NewTask] for a new task


class HealthResponse(BaseModel):
    status: bool


class MachineLearningDataInput(BaseModel):
    image_url: str

    def get_image(self):
        """
        Gets the image from the given URL.
        """
        try:
            with urlopen(self.image_url) as response:
                image_bytes = response.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            return pil_image
        except HTTPException as e:
            # Handle the exception here
            print(f"Error reading image: {e}")
            return None
