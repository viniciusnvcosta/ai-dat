import json

from core.config import settings
from core.setup import create_application
from magnum import Magnum

"""
    The FastAPI application setup.
    This module creates and configures the FastAPI application
    based on the provided settings to deploy the App.
"""

app = create_application(settings=settings)

magnum_handler = Magnum(app)


def lambda_handler(event, context):
    # Call the Magnum handler with the event and context
    return magnum_handler(event, context)
