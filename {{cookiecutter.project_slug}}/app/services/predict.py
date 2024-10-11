import os
from typing import Any, Dict, List

import cv2
import numpy as np
import tensorflow as tf
from core.config import settings
from core.errors import ModelLoadException
from loguru import logger
from PIL import Image
from ultralytics import YOLO

from services.process import ResultProcessorRunner as rs_processor

# * uncomment the following line to import the preprocessing class
# from ml.model.image_processor import ImageProcessor as processor
from ml.model.model_runner import (
    TensorflowClassifier,
    YoloClassifier,
    YoloDetector,
)


class MachineLearningModelHandlerScore(object):
    """
    Class for handling machine learning models and making predictions.

    Methods:
        predict(image_input: Image.Image, load_wrapper: Any, method: str = "predict", model_type: str = "classifier") -> Any:
            Predicts the output of the machine learning model based on the given image input.

            Args:
                image_input (Image.Image): The input image for prediction.
                load_wrapper (Any): The load wrapper function or object used to load the model.
                method (str, optional): The prediction method to use. Defaults to "predict".
                model_type (str, optional): The type of model to use. Defaults to "classifier".

            Returns:
                Any: The final prediction score.

        get_runner(load_wrapper, model_type: str = "classifier") -> object:
            Returns the appropriate runner class based on the load wrapper and model type.

            Args:
                load_wrapper (Any): The load wrapper function or object used to load the model.
                model_type (str, optional): The type of model to use. Defaults to "classifier".

            Returns:
                object: The runner class.

        get_model(load_wrapper: Any) -> Any:
            Returns the loaded machine learning model.

            Args:
                load_wrapper (Any): The load wrapper function or object used to load the model.

            Returns:
                Any: The loaded machine learning model.

        load(load_wrapper: Any) -> Any:
            Loads the machine learning model using the given load wrapper.

            Args:
                load_wrapper (Any): The load wrapper function or object used to load the model.

            Returns:
                Any: The loaded machine learning model.
    """

    model = None

    @classmethod
    def predict(
        cls,
        image_input: Image.Image,
        load_wrapper: Any,
        method: str = "predict",
        model_type: str = "classifier",
    ) -> List[Dict]:
        model = cls.get_model(load_wrapper)

        cv2_input = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

        # Dynamically select the runner
        runner_class = cls.get_runner(load_wrapper, model_type)
        # Load the model to the runner
        logger.debug("runner_class: ", runner_class.__name__)
        runner = runner_class(model=model)

        try:
            # note: if final_score is equal to result,
            # then the runner_class does not have a process method
            result = getattr(runner, method)(cv2_input)
            # get input image shape
            image_shape = cv2_input.shape[:2]
            logger.debug(f"og_img_size: {image_shape, type(image_shape)}")

            final_score = rs_processor.process(
                runner_class.__name__, result, image_shape
            )
        except Exception as err:
            raise ValueError(f"Error during prediction: {err}") from err

        return final_score

    @staticmethod
    def get_runner(load_wrapper, model_type: str = "classifier") -> object:
        runner_map = {
            YOLO: {
                "detector": YoloDetector,
                "classifier": YoloClassifier,
            },
            tf.keras.models.load_model: {
                "classifier": TensorflowClassifier,
            },
            # Add future models here
        }

        if load_wrapper not in runner_map or model_type not in runner_map[load_wrapper]:
            raise ValueError(
                f"Unsupported load wrapper or model type: {load_wrapper}, {model_type}"
            )

        return runner_map[load_wrapper][model_type]

    @classmethod
    def get_model(cls, load_wrapper: Any):
        if cls.model is None and load_wrapper:
            cls.model = cls.load(load_wrapper)
        return cls.model

    @staticmethod
    def load(load_wrapper: Any):
        path = os.path.join(settings.MODEL_PATH, settings.MODEL_NAME)
        if not os.path.exists(path):
            message = f"Machine learning model at {path} does not exist!"
            logger.error(message)
            raise FileNotFoundError(message)

        model = load_wrapper(path)
        if model is None:
            message = f"Model could not be loaded from {path}!"
            logger.error(message)
            raise ModelLoadException(message)

        return model
