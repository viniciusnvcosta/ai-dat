import pickle as pkl
from pathlib import Path

import click
from loguru import logger

import ml.train as train
from app.core.config import EnvironmentSettings
from app.services.predict import MachineLearningModelHandlerScore as handler

settings = EnvironmentSettings()


def evaluate_model(model: object, dataset: bool = True):
    """
    Evaluate the model using the dev or test dataset.

    Parameters
    ----------
    dev : bool, optional
        Whether to evaluate the model on the dev dataset, by default True

    Returns
    -------
    cache : dict
        A dictionary containing the evaluation metrics.
    """
    cache = dict(
        predictions=None,
        ground_truth=None,
    )

    # Evaluate the model
    # + Apply pipeline steps
    cache = None

    return cache


@click.command()
@click.option(
    "--from_path",
    default=f"{settings.MODEL_PATH}{settings.MODEL_NAME}",
    type=click.Path(),
    help="Path to load the trained model.",
)
@click.option(
    "--dev",
    default=True,
    type=bool,
    help="Whether to evaluate the model on the dev dataset.",
)
def main(from_path, dev):
    """
    This module is used to evaluate the model.

    It loads the trained model and evaluates its performance on the dev or test dataset.
    """
    logger.info(f"Evaluating the model from {from_path}.")

    # Load the model
    model = handler.load(load_wrapper=pkl.load)

    eval_metrics = evaluate_model(model, dev)

    # Save the evaluation metrics
    logger.success(f"Evaluation metrics: {eval_metrics}")
