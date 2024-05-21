import io
import json

from core.config import settings
from fastapi import APIRouter, File, HTTPException, UploadFile
from models.prediction import (
    HealthResponse,
    MachineLearningDataInput,
    MachineLearningResponse,
)
from PIL import Image
from services.predict import MachineLearningModelHandlerScore as model
from services.utils import Utils as utils

router = APIRouter()  # Creating an APIRouter instance


# Function to get predictions using the ML model
def get_prediction(data_point):
    return model.predict(data_point, load_wrapper=utils.loader())


# Endpoint for handling POST requests to '/predict' route
@router.post(
    "/predict",
    response_model=MachineLearningResponse,
    name="predict:get-data",
)
async def predict(image_file: UploadFile = File(...)):
    if not image_file:
        raise HTTPException(status_code=404, detail="'image_file' argument invalid!")
    try:
        image_data = await image_file.read()
        PIL_image = Image.open(io.BytesIO(image_data))
        final_score = get_prediction(PIL_image)
        return MachineLearningResponse(result=final_score)

    except Exception as err:  # Handling exceptions
        raise HTTPException(
            status_code=500, detail=f"Exception: {err}"
        )  # Raising HTTP exception with error details


# Endpoint for handling GET requests to '/health' route
@router.get(
    "/test",
    response_model=MachineLearningResponse,
    name="test:get-data",
)
async def test():
    # is_health = False
    try:
        # Loading an example image from configuration files
        with open(settings.INPUT_EXAMPLE, "rb") as image_file:
            image_data = image_file.read()  # Reading image file data
        # Opening image with PIL
        PIL_image = Image.open(io.BytesIO(image_data))
        # Getting prediction to verify if the prediction system is functioning correctly
        test_prediction = get_prediction(PIL_image)
        # is_health = True  # If no exceptions, system is considered healthy
        return MachineLearningResponse(
            result=test_prediction
        )  # Returning health status

    except Exception as err:  # Handling exceptions
        raise HTTPException(
            status_code=404, detail=f"Unhealthy: {err}"
        )  # Raising HTTP exception for unhealthy status


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
