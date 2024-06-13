from typing import Callable
import PeftModel
from loguru import logger

from fastapi import FastAPI


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

    MachineLearningModelHandlerScore.get_model(load_wrapper=PeftModel.from_pretrained)


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