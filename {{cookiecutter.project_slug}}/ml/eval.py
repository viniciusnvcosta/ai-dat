"""This module is used to evaluate the trained model.

    It utilizes the training pipeline to evaluate the model's performance with a dev or test dataset.

    Performs metrics evaluation and save the results for further analysis.

"""
from ml.model.pipeline import ...
from ml.train import ...
from app.core.config import EnvironmentSettings

settings = EnvironmentSettings()

