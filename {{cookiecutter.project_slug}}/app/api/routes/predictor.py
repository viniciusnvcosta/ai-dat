import json

from core.config import settings
from fastapi import APIRouter, HTTPException
from models.prediction import (
    MachineLearningDataInput,
    MachineLearningResponse,
)
from PIL import Image
from services.predict import MachineLearningModelHandlerScore as model
from services.utils import Utils

router = APIRouter()  # Creating an APIRouter instance


# Function to get predictions using the ML model
def get_prediction(data_point):
    return model.predict(data_point)


# Endpoint for handling POST requests to '/predict' route
@router.post(
    "/predict",
    response_model=MachineLearningResponse,
    name="predict:get-data",
)
async def predict(text_input: MachineLearningDataInput):

    if not text_input:
        raise HTTPException(status_code=404, detail="'text_input' argument invalid!")
    try:
        text_point = text_input.get_prompt()
        prediction = get_prediction(text_point)

        # CIDs = Utils.extract_CID(prediction)

        result = {
            "reponse_text": prediction,
            # "cid_groups": CIDs,
        }
        return MachineLearningResponse(result=result)
    # Handling Exceptions
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except ValueError as err:
        raise HTTPException(status_code=422, detail=f"Unprocessable Entity: {err}")
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {err}")


# Endpoint for handling GET requests to a 'health'-like route
@router.get(
    "/test",
    response_model=MachineLearningResponse,
    name="test:get-data",
)
async def test():
    is_health = False
    try:
        text_input = MachineLearningDataInput(
            **json.loads(open(settings.INPUT_EXAMPLE, "r").read())
        )
        text_point = text_input.get_prompt()
        prediction = get_prediction(text_point)

        # CIDs = Utils.extract_CID(prediction)

        is_health = True
        result = {
            "reponse_text": prediction,
            # "cid_groups": CIDs,
            "is_healthy": is_health,
        }
        return MachineLearningResponse(result=result)
    # Returning unhealth status
    except Exception as err:
        raise HTTPException(status_code=404, detail=f"Unhealthy: {err}")
