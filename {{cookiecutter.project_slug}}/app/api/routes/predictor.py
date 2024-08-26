import io
import json

from core.config import settings
from fastapi import APIRouter, File, HTTPException, UploadFile
from models.prediction import (
    HealthResponse,
    MachineLearningDataInput,
    MachineLearningResponse,
    ClassificationResult,
)
from PIL import Image
from services.predict import MachineLearningModelHandlerScore as model
from services.utils import Utils as utils

router = APIRouter()  # Creating an APIRouter instance


def get_prediction(image_point: Image.Image) -> ClassificationResult:
    try:
        predict_result = model.predict(image_point)
        data = ClassificationResult(
            # Replace with actual keys and values
            class_name=predict_result["class_name"],
            classification_score=predict_result["classification_score"],
        )
        return data
    except ValueError as e:
        print(f"An error occurred during prediction: {e}")


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
        if body.image_url == "":
            return MachineLearningResponse(
                status_code=400, content={"msg": "image_url is required"}
            )
        input_image = body.get_image()

        result = get_prediction(input_image)
        return MachineLearningResponse(status_code=200, content=result.model_dump())

    except Exception as err:  # Handling exceptions
        raise HTTPException(
            status_code=500, detail=f"Exception: {err}"
        ) from err  # Raising HTTP exception with error details


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
