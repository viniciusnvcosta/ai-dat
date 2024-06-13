import os

import torch
from core.config import settings
from core.errors import ModelLoadException, PredictException
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class MachineLearningModelHandlerScore(object):
    model = None

    @classmethod
    def predict(
        cls, input, load_wrapper=None, method="generate"
    ):  # defaul method is predict
        llm = cls.get_model(load_wrapper)

        tokenizer = cls.load_tokenizer()

        # Tokenize the input
        try:
            inputs = tokenizer(input, return_tensors="pt")
        except Exception as e:
            raise PredictException(f"Error in tokenizing input: {str(e)}")

        kwargs = {
            "max_new_tokens": settings.MAX_NEW_TOKENS,
            "use_cache": settings.USE_CACHE,
        }

        # Check if the method exists in the model and call it with the appropriate arguments
        if hasattr(llm, method):
            try:
                output_ids = getattr(llm, method)(**inputs, **kwargs)
                output_text = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                return output_text
            except Exception as e:
                raise PredictException(f"Error in generating output: {str(e)}")
        else:
            raise PredictException(f"'{method}' attribute is missing")

    @classmethod
    def get_model(cls, load_wrapper):
        if cls.model is None and load_wrapper:
            cls.model = cls.load(load_wrapper)
        return cls.model

    @classmethod
    def load_tokenizer(cls):
        tokenizer = None
        if settings.MODEL_PATH.endswith("/"):
            path = f"{settings.MODEL_PATH}{settings.TOKENIZER_NAME}"
        else:
            path = f"{settings.MODEL_PATH}/{settings.TOKENIZER_NAME}"

        tokenizer = AutoTokenizer.from_pretrained(path)

        if not tokenizer:
            message = f"Tokenizer {tokenizer} could not load!"
            logger.error(message)
            raise ModelLoadException(message)
        return tokenizer

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
        # model = load_wrapper(path)
        # TODO
        model = load_wrapper(path)

        if not model:
            message = f"Model {model} could not load!"
            logger.error(message)
            raise ModelLoadException(message)
        return model
