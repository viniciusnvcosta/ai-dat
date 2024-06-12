from typing import Callable

from fastapi import FastAPI
from loguru import logger

# * import any model loading code here as a wrapper function

# def model_settings():
#     """
#     Load model settings
#     """
#     from core.config import ModelSettings

#     return ModelSettings()

def preload_model():
    """
    In order to load model on memory to each worker
    """
    from services.predict import MachineLearningModelHandlerScore

    # * add load function here
    MachineLearningModelHandlerScore.get_model(load_wrapper=any())


def create_start_app_handler(app: FastAPI) -> Callable:
    def start_app() -> None:
        preload_model()
        logger.info("Application started")

    return start_app

def create_stop_app_handler(app: FastAPI) -> Callable:
    def stop_app() -> None:
        # drop app resources
        logger.info("Application stopped")
        pass

    return stop_app