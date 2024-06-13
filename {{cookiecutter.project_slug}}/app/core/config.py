import logging
import os
import sys
from enum import Enum
from typing import List, Optional, Union, Any

from loguru import logger
from core.logging import InterceptHandler
from pydantic_settings import BaseSettings
from starlette.config import Config
from starlette.datastructures import Secret

current_file_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(current_file_dir, "..", "..", ".env")
config = Config(env_path)


class AppSettings(BaseSettings):
    APP_NAME: str = config("PROJECT_NAME", default="insert_name-ia")
    APP_DESCRIPTION: str | None = config("APP_DESCRIPTION", default=None)
    APP_VERSION: str | None = config("APP_VERSION", default="0.1.0")
    API_PREFIX: str | None = config("API_PREFIX", default="/api")
    DEBUG: bool = config("DEBUG", cast=bool, default=False)
    CONTACT_NAME: str | None = config("CONTACT_NAME", default=None)
    CONTACT_EMAIL: str | None = config("CONTACT_EMAIL", default=None)
    LICENSE_NAME: str | None = config("LICENSE", default=None)
    # logging configuration
    LOGGING_LEVEL: int = logging.DEBUG if DEBUG else logging.INFO  # Add type annotation
    logging.basicConfig(
        handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
    )
    logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])


class CryptSettings(BaseSettings):
    SECRET_KEY: Secret = config("SECRET_KEY", cast=Secret, default="")
    ALGORITHM: str = config("ALGORITHM", default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=30)
    REFRESH_TOKEN_EXPIRE_DAYS: int = config("REFRESH_TOKEN_EXPIRE_DAYS", default=7)


class EnvironmentOption(Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class EnvironmentSettings(BaseSettings):
    # ENVIRONMENT: EnvironmentOption = config("ENVIRONMENT", default="local")
    MODEL_PATH: str = config("MODEL_PATH", default="./ml/model/")
    MODEL_NAME: str = config("MODEL_NAME", default="lora_model")
    TOKENIZER_NAME: str = config("TOKENIZER_NAME", default="lora_model")
    INPUT_EXAMPLE: str = config(
        "INPUT_EXAMPLE", default="./ml/model/examples/example.json"
    )


class ModelSettings(BaseSettings):
    MAX_NEW_TOKENS: int = config("MAX_NEW_TOKENS", cast=int, default=200)
    USE_CACHE: bool = config("USE_CACHE", cast=bool, default=True)


#     # Unsloth Llama 3
#     MAX_SEQ_LENGTH: int = config("MAX_SEQ_LENGTH", cast=int, default=3000)
#     DTYPE: Optional[str] = config("DTYPE", default=None)
#     LOAD_IN_4BIT: bool = config("LOAD_IN_4BIT", cast=bool, default=True)
#     # LoraConfig
#     ALPHA_PATTERN: dict = config("ALPHA_PATTERN", default={""})
#     AUTO_MAPPING: Optional[Any] = config("AUTO_MAPPING", default=None)
#     BASE_MODEL_NAME_OR_PATH: str = config("BASE_MODEL_NAME_OR_PATH", default="unsloth/llama-3-8b-bnb-4bit")
#     R: int = config("R", cast=int, default=16)
#     TARGET_MODULES: List[str] = config("TARGET_MODULES", cast=List[str], default=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ])
#     LORA_ALPHA: int = config("LORA_ALPHA", cast=int, default=16)
#     LORA_DROPOUT: float = config("LORA_DROPOUT", cast=float, default=0)
#     FAN_IN_FAN_OUT: str = config("FAN_IN_FAN_OUT", default="none")
#     BIAS: str = config("BIAS", default="none")
#     RANK_PATTERN: dict = config("RANK_PATTERN", default={""})
#     USE_GRADIENT_CHECKPOINTING: Union[bool, str] = config("USE_GRADIENT_CHECKPOINTING", default="unsloth")
#     USE_DORA: bool = config("USE_DORA", cast=bool, default=False)
#     USE_RSLORA: bool = config("USE_RSLORA", cast=bool, default=False)
#     RANDOM_STATE: int = config("RANDOM_STATE", cast=int, default=3407)
#     LOFTQ_CONFIG: Optional[Any] = config("LOFTQ_CONFIG", default=None)
#     # PeftConfig
#     PEFT_TYPE: "str" = config("PEFT_TYPE", default="LORA")
#     TASK_TYPE: "str" = config("TASK_TYPE", default="CAUSAL_LM")
#     INFERENCE_MODE: bool = config("INFERENCE_MODE", cast=bool, default=True)


# class GPTSettings(BaseSettings):
#     GPT_API_KEY: str = config("GPT_API_KEY", default="")
#     GPT_MODEL: str = config("GPT_MODEL", default="gpt-3.5-turbo-16k")
#     GPT_TEMPERATURE: float = config("GPT_TEMPERATURE", default=0)
#     GPT_PDF_DEFAULT_QUESTION: str = config("GPT_PDF_DEFAULT_QUESTION", default="")
#     GPT_PDF_PAGE_PARTS: int = config("GPT_PDF_PAGE_PARTS", default=5)
#     GPT_STATUS_COLOR_QUESTION: str = config("GPT_STATUS_COLOR_QUESTION", default="")
#     GPT_STATUS_COLOR_INDEX_COMPARE_TABLE_PATH: str = config(
#         "GPT_STATUS_COLOR_INDEX_COMPARE_TABLE_PATH", default=""
#     )
#     GPT_STATUS_COLOR_SINOUNS_JSON_PATH: str = config(
#         "GPT_STATUS_COLOR_SINOUNS_JSON_PATH", default=""
#     )
#     GPT_STATUS_COLOR_DICT_BASE_JSON_PATH: str = config(
#         "GPT_STATUS_COLOR_DICT_BASE_JSON_PATH", default=""
#     )
#     GPT_GROUP_COLORS_JSON_PATH: str = config("GPT_GROUP_COLORS_JSON_PATH", default="")


class S3ConnectionSettings(BaseSettings):
    S3_BUCKET_NAME: str = config("S3_BUCKET_NAME", default="")
    S3_ACCESS_KEY: str = config("S3_ACCESS_KEY", default="")
    S3_SECRET_KEY: str = config("S3_SECRET_KEY", default="")


# ? not sure if needed in future
# class DefaultRateLimitSettings(BaseSettings):
#     DEFAULT_RATE_LIMIT_LIMIT: int = config("DEFAULT_RATE_LIMIT_LIMIT", default=10)
#     DEFAULT_RATE_LIMIT_PERIOD: int = config("DEFAULT_RATE_LIMIT_PERIOD", default=3600)
#     MAX_CONNECTIONS_COUNT: int = config("MAX_CONNECTIONS_COUNT", cast=int, default=10)
#     MIN_CONNECTIONS_COUNT: int = config("MIN_CONNECTIONS_COUNT", cast=int, default=10)


class Settings(
    AppSettings,
    CryptSettings,
    EnvironmentSettings,
    ModelSettings,
    # GPTSettings,
    S3ConnectionSettings,
):
    pass


# use this to import settings in other modules
settings = Settings()
