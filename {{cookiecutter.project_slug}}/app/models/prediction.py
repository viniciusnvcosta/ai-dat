from pydantic import BaseModel
from typing import List


class DetectionResult(BaseModel):
    bbox: List[int]
    class_name: str
    detection_score: float


class ClassificationResult(BaseModel):
    class_name: int
    classification_score: float


class MachineLearningResponse(BaseModel):
    result: List[
        DetectionResult
    ]  # Change to List[ClassificationResult] if classification task or List[NewTask] for a new task


class HealthResponse(BaseModel):
    status: bool


class MachineLearningDataInput(BaseModel):
    image_file: str
