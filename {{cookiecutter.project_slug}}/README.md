# {{cookiecutter.project_name}}

{{cookiecutter.project_short_description}}

## Development Requirements

- Python3.11
- Pip
- Poetry (Python Package Manager)
- FastAPI
- Pydantic v2
- Pydantic Settings v2

### M.L Model Environment

```sh
MODEL_PATH=./ml/model/
MODEL_NAME=model.pt
```

### Update `/predict`

To update your machine learning model, match your model's `load wrapper` for binaries (e.g. [load](https://joblib.readthedocs.io/en/latest/generated/joblib.load.html) from joblib), `method` (e.g. [predict](https://keras.io/api/models/model_training_apis/#predict-method) from keras) dependending on what ML framework is used and `model_type` (_detector_ or _classifier_) at `predictor.py` pipeline function `get_prediction()` [here](app/api/routes/predictor.py#L23).

These parameters are needed to call the correct `runner` class for the model.

```python
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
```

#### Note

> The custom `runner` classes are model-specific handlers for loading and running the models. The featured handlers are mapped [here](app/services/predict.py#L107) at `predict.py` on the static method `get_runner()`.

## Installation

```sh
python -m venv venv
source venv/bin/activate
make install
```

## Running on Localhost

`make run`

## Build app for deployment

`make deploy`

## Running Tests

`make test`

## Access Swagger Documentation

> <http://localhost:8080/docs>

## Access Redocs Documentation

> <http://localhost:8080/redoc>

## Project structure

Files related to application are in the `app` or `tests` directories.
Application parts are:

```plaintext
    app
    │
    │ # Fast-API stuff
    ├── api                   - web related stuff.
    │   └── routes            - web routes.
    ├── core                  - application configuration, startup events, logging.
    ├── models                - pydantic models for this application.
    ├── services              - logic that is not just crud related.
    │   ├── __init__.py       - makes services a Python module
    │   ├── predict.py        - Predictor class for model loading and prediction.
    │   └── utils.py          - utility functions
    │
    ├── lambda_function.py    - [Main] FastAPI application for AWS Lambda creation and configuration.
    └── main.py               - [Main] FastAPI application creation with gunicorn server.
    │
    │
    │ # ML stuff
    ├── data               - where you persist data locally
    │   ├── interim        - intermediate data that has been transformed.
    │   ├── processed      - the final, canonical data sets for modeling.
    │   └── raw            - the original, immutable data dump.
    │
    ├── notebooks          - Jupyter notebooks. Naming convention is a sprint number (for ordering).
    │
    ├── ml                 - modelling source code for use in this project.
    │   ├── __init__.py    - makes ml a Python module
    │   ├── train.py       - training script
    │   ├── eval.py        - evaluation script
    │   │
    │   ├── data           - scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       - scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── model          - model binaries and inference pipeline scripts
    │       ├── image_processor.py
    │       ├── model_runner.py
    │       └── result_runner.py
    │
    │
    └── tests              - pytest
    │
    ├── Dockerfile         - The Dockerfile to build the container image.
    ├── docker-compose.yml - The Docker Compose file to run the application and its dependencies.
    ├── lambda_dockerfile  - The Dockerfile to build the container image for AWS Lambda.
    ├── Makefile           - Makefile for build and deployment commands.
    ├── template.yaml      - A template that defines the SAM application's AWS resources.
    └── requirements.txt   - The pip requirements to be installed during the container build.
```

## GCP

Deploying inference service to Cloud Run

### Authenticate to GCP

1. Install `gcloud` cli
2. `gcloud auth login`
3. `gcloud config set project <PROJECT_ID>`

### Enable APIs

1. Cloud Run API
2. Cloud Build API
3. IAM API

### Deploy to Cloud Run

1. Run `gcp-deploy.sh`

### Clean up

1. Delete Cloud Run
2. Delete Docker image in GCR

## AWS

Deploying inference service to AWS Lambda

### Authenticate to AWS

1. Install `awscli` and `sam-cli`
2. `aws configure`

### Deploy to Lambda

1. Run `sam build`
2. Run `sam deploy --guiChange this portion for other types of models

## Add the correct type hinting when completed

`aws cloudformation delete-stack --stack-name <STACK_NAME_ON_CREATION>`
