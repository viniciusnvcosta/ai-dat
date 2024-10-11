""" Application configuration settings """

import logging
import os
import sys
from enum import Enum
from typing import Optional, Union

from core.logging import InterceptHandler
from loguru import logger
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the path to the .env file
current_file_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(current_file_dir, "..", "..", ".env")


class AppSettings(BaseSettings):
    APP_NAME: str = "insert_name-ia"
    APP_DESCRIPTION: Optional[str] = None
    APP_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    CONTACT_NAME: Optional[str] = None
    CONTACT_EMAIL: Optional[str] = None
    LICENSE_NAME: Optional[str] = None


class EnvironmentOption(str, Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class CryptSettings(BaseSettings):
    SECRET_KEY: SecretStr = Field(..., env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7


class EnvironmentSettings(BaseSettings):
    ENVIRONMENT: EnvironmentOption = EnvironmentOption.LOCAL
    MODEL_PATH: str = "./ml/model/"
    MODEL_NAME: str = "best.pt"
    INPUT_EXAMPLE: str = "./ml/model/examples/example.json"


class ModelSettings(BaseSettings):
    IMG_SIZE: int = 640
    CONF_THRESHOLD: float = 0.501
    IOU_THRESHOLD: float = 0.45


class S3ConnectionSettings(BaseSettings):
    S3_BUCKET_NAME: str = ""
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_KEY: SecretStr = Field(..., env="S3_SECRET_KEY")


# ? not implemented
# class DefaultRateLimitSettings(BaseSettings):
#     DEFAULT_RATE_LIMIT_LIMIT: int = 10
#     DEFAULT_RATE_LIMIT_PERIOD: int = 3600
#     MAX_CONNECTIONS_COUNT: int = 10
#     MIN_CONNECTIONS_COUNT: int = 10


class Settings(
    AppSettings,
    EnvironmentSettings,
    ModelSettings,
    S3ConnectionSettings,
):
    # Specify the env_file in the model config
    model_config = SettingsConfigDict(env_file=env_path, extra="ignore")


# Instantiate settings
settings = Settings()

# Configure logging after settings are loaded
LOGGING_LEVEL = logging.DEBUG if settings.DEBUG else logging.INFO
logging.basicConfig(
    handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
)
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])
