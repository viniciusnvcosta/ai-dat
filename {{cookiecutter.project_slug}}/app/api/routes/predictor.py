import json

import joblib
from fastapi import APIRouter, HTTPException

from core.config import settings
from services.predict import MachineLearningModelHandlerScore as model
from models.prediction import (
    HealthResponse,
    MachineLearningResponse,
    MachineLearningDataInput,
)

router = APIRouter()


## Change this portion for other types of models
## Add the correct type hinting when completed
def get_prediction(data_point):
    return model.predict(data_point, load_wrapper=joblib.load, method="predict")


@router.post(
    "/predict",
    response_model=MachineLearningResponse,
    name="predict:get-data",
)
async def predict(data_input: MachineLearningDataInput):

    if not data_input:
        raise HTTPException(status_code=404, detail="'data_input' argument invalid!")
    try:
        data_point = data_input.get_np_array()
        prediction = get_prediction(data_point)

        # * Define the response object outputs here (as a dictionary)
        result = {"prediction": prediction}
        return MachineLearningResponse(prediction=result)
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
        test_input = MachineLearningDataInput(
            **json.loads(open(settings.INPUT_EXAMPLE, "r").read())
        )
        test_point = test_input.get_np_array()
        prediction = get_prediction(test_point)

        is_health = True
        result = {
            "predicted_label": prediction,
            "is_healthy": is_health,
        }
        return HealthResponse(status=result)
    except Exception:
        raise HTTPException(status_code=404, detail="Unhealthy")
