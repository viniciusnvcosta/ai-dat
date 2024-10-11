import io
import json
from typing import Any, Dict, List

from core.config import settings
from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger
from models.prediction import (
    # ClassificationResult,
    DetectionResult,
    HealthResponse,
    MachineLearningDataInput,
    MachineLearningResponse,
)
from PIL import Image
from services.predict import MachineLearningModelHandlerScore as model
from services.utils import Utils as utils
from ultralytics import YOLO

router = APIRouter()  # Creating an APIRouter instance


def get_prediction(image_point: Image.Image) -> List[DetectionResult]:
    """
    Calls the model's prediction pipeline on the input image and returns the results.
    Args:
        image_point (Image.Image): The input image for which predictions are to be made.
    Returns:
        List[DetectionResult]: A list of detection results, each containing class name, detection score, and bounding box.

        This may as well be a list of ClassificationResult objects if the model is a classifier.
    Raises:
        HTTPException: If the prediction process fails, an HTTPException is raised with a status code of 400 and an error message.
    """

    try:
        predict_results: List[Dict[str, Any]] = model.predict(
            image_point,
            # * Change the function call arguments to match the model's prediction
            load_wrapper=YOLO,
            method="inference",
            model_type="detector",
        )

        # * Change the following code to match the model's task
        detection_results = [
            DetectionResult(
                # Replace with actual keys and values
                class_name=ix["class_name"],
                detection_score=ix["detection_score"],
                bbox=ix["bbox"],
            )
            for ix in predict_results
        ]

        return detection_results

    except ValueError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


# Endpoint for handling POST requests to '/predict' route
@router.post(
    "/predict",
    response_model=MachineLearningResponse,
    name="predict:get-data",
)
async def predict(body: MachineLearningDataInput):
    try:
        # if not body:
        #     raise HTTPException(status_code=400, detail="Invalid request!")
        if body.image == "":
            return MachineLearningResponse(
                status_code=400, result={"msg": "image is required"}
            )
        input_image = body.get_image_base64()

        result = get_prediction(input_image)
        logger.debug(f"1st result check: {result[0].model_dump()}")
        logger.info("Prediction completed")
        return MachineLearningResponse(status_code=200, result=result)

    except Exception as err:  # Handling exceptions
        if isinstance(err, HTTPException):
            raise err
        logger.exception("An error occurred during prediction.")
        raise HTTPException(status_code=500, detail=f"Exception: {err}").with_traceback(
            err.__traceback__
        )  # Raising HTTP exception with error details and traceback


# Endpoint for handling GET requests to '/health' route
@router.get(
    "/test",
    response_model=MachineLearningResponse,
    name="test:get-data",
)
async def test():
    try:
        # Loading an example image request from a test file
        with open(settings.INPUT_EXAMPLE, "r", encoding="utf-8") as file:
            request_data = json.load(file)
        body = MachineLearningDataInput(**request_data)

        if not body.get_image():
            raise ValueError("Failed to read image")

        # If everything is successful, mark the service as healthy
        result = {
            "is_healthy": True,
        }
        return HealthResponse(result=result)

    except Exception as err:  # Handling exceptions
        raise HTTPException(
            status_code=404, detail=f"Unhealthy: {err}"
        ) from err  # Raising HTTP exception for unhealthy status


# @router.get(
#     "/health",
#     response_model=HealthResponse,
#     name="health:get-data",
# )
# async def health():
#     is_health = False
#     try:
#         test_input = MachineLearningDataInput(
#             **json.loads(open(settings.INPUT_EXAMPLE, "r").read())
#         )
#         test_point = test_input.get_np_array()
#         get_prediction(test_point)
#         is_health = True
#         return HealthResponse(status=is_health)
#     except Exception:
#         raise HTTPException(status_code=404, detail="Unhealthy")
