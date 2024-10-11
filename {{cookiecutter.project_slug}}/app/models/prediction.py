import base64
from io import BytesIO
from typing import Any, List
from urllib.request import urlopen

import numpy as np
from fastapi import HTTPException
from loguru import logger
from PIL import Image
from pydantic import BaseModel


class DetectionResult(BaseModel):
    class_name: str
    detection_score: float
    bbox: List[int]


class ClassificationResult(BaseModel):
    class_name: int
    classification_score: float


class MachineLearningResponse(BaseModel):
    status_code: int
    result: List[DetectionResult]
    # * result: List[DetectionResult] or List[ClassificationResult] if classification task or List[NewTask] for a new task


class HealthResponse(BaseModel):
    status: bool


class MachineLearningDataInput(BaseModel):
    # image_url: str
    image: str

    def get_image_base64(self):
        """
        Gets the image from the given URL and converts it to base64.
        """
        try:
            image_bytes = base64.b64decode(self.image)
            pil_image = Image.open(BytesIO(image_bytes))
            return pil_image
        except HTTPException as e:
            # Handle the exception here
            logger.error(f"Error reading image: {e}")
            return None

    def get_image_url(self):
        """
        Gets the image from the given URL.
        """
        try:
            with urlopen(self.image_url) as response:
                image_bytes = response.read()
            pil_image = Image.open(BytesIO(image_bytes))
            return pil_image
        except HTTPException as e:
            # Handle the exception here
            logger.error(f"Error reading image: {e}")
            return None
