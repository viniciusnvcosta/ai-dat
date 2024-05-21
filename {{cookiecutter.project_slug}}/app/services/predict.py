import os
from fastapi import HTTPException
import cv2
import numpy as np
import tensorflow as tf
from core.config import settings
from core.errors import ModelLoadException
from loguru import logger
from ultralytics import YOLO

# * uncomment the following line to import the preprocessing class
# from ml.model.image_processor import ImageProcessor as transformer
from ml.model.model_runner import TensorflowClassifier, YoloDetector
from ml.model.result_runner import YoloDetectorResult


class MachineLearningModelHandlerScore(object):
    model = None

    @classmethod
    def predict(cls, input, load_wrapper=None):
        # Import model
        model = cls.get_model(load_wrapper)
        # Perform prediction
        final_score = cls.pipeline(input, load_wrapper, model)
        print(final_score)
        return final_score

    @classmethod
    def get_model(cls, load_wrapper):
        if cls.model is None and load_wrapper:
            cls.model = cls.load(load_wrapper)
        return cls.model

    @staticmethod
    def load(load_wrapper):
        model = None
        if settings.MODEL_PATH.endswith("/"):
            path = f"{settings.MODEL_PATH}{settings.MODEL_NAME}"
        else:
            path = f"{settings.MODEL_PATH}/{settings.MODEL_NAME}"
        if not os.path.exists(path):
            message = f"Machine learning model at {path} not exists!"
            logger.error(message)
            raise FileNotFoundError(message)
        model = load_wrapper(path)
        if not model:
            message = f"Model {model} could not load!"
            logger.error(message)
            raise ModelLoadException(message)

        return model

    @staticmethod
    def pipeline(input, load_wrapper, model):
        """
        This method runs all the inference and prediction logic for the AI model.
        """
        # PIL to cv2
        input = cv2.cvtColor(np.array(input), cv2.COLOR_RGB2BGR)
        # * Preprocess the image if necessary (Check ImageProcessor class for more options)
        # input = transformer.filter_image(input)
        # Perform prediction based on the model
        if load_wrapper == YOLO:
            try:
                # Perform prediction
                result = YoloDetector(model).inference(
                    input
                )  # Use YoloClassifier for classification tasks
                # Get the prediction scores
                final_score = YoloDetectorResult().run_scoring_result(
                    result
                )  # Use YoloScoringResult for classification tasks
            except Exception as err:  # Handling exceptions
                raise ValueError(
                    f"Unsupported model format: YOLO model not supported, use .tf models for classification tasks.{err}"
                )

        elif load_wrapper == tf.keras.models.load_model:
            try:
                # Perform prediction
                final_score = TensorflowClassifier(model).inference(input)
            except Exception as err:
                raise ValueError(
                    f"Unsupported model format: Tensorflow model not supported, use .pt models for detection tasks. {err}"
                )

        return final_score
