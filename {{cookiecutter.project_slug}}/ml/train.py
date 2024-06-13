import pickle as pkl

import click
from loguru import logger

import ml.data as data
import ml.features as features
from app.core.config import EnvironmentSettings

settings = EnvironmentSettings()


def pipeline():
    logger.info("Start training the model.")

    # + Implement training loop
    model = None

    return model


@click.command()
@click.argument(
    "--model_path",
    default=settings.MODEL_PATH,
    type=click.Path(),
    help="Path to save the trained model.",
)
def main(model_path):
    """
    This module is used to train the model.

    It utilizes the feature and data module pipelines to perform the training.

    Performs the training and saves any given model to the specified path location.
    """
    logger.info(f"Training the model and saving to {model_path}.")

    model = pipeline()

    # Save the model
    logger.success(f"Model saved to {model_path}.")


if __name__ == "__main__":
    main()
