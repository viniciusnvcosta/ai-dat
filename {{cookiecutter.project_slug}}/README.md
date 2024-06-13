# pickcells-mcids-ia

Repositório da IA de mcids estruturada no padrão PCP

## Development Requirements

- Python3.11.0
- Pip
- Poetry (Python Package Manager)

### M.L Model Environment

```sh
MODEL_PATH=./ml/model/
MODEL_NAME=model.pkl
```

### Update `/predict`

To update your machine learning model, add your `load` and `method` [change here](app/api/routes/predictor.py#L19) at `predictor.py`

## Installation

```sh
python -m venv venv
source venv/bin/activate
make install
```

## Runnning Localhost

`make run`

## Deploy app

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

    app
    │
    │ # Fast-API stuff
    ├── api                 - web related stuff.
    │   └── routes          - web routes.
    ├── core                - application configuration, startup events, logging.
    ├── models              - pydantic models for this application.
    ├── services            - logic that is not just crud related.
    │   ├── __init__.py     - makes services a Python module
    │   ├── predict.py      - Predictor class for model loading and prediction.
    │   └── utils.py        - utility functions
    │
    ├── lambda_function.py  - [Main] FastAPI application for AWS Lambda creation and configuration.
    └── main.py             - [Main] FastAPI application creation with gunicorn server.
    │
    │ # ML stuff
    ├── data             - where you persist data locally
    │   ├── interim      - intermediate data that has been transformed.
    │   ├── processed    - the final, canonical data sets for modeling.
    │   └── raw          - the original, immutable data dump.
    │
    │
    ├── notebooks        - Jupyter notebooks. Naming convention is a sprint number (for ordering),
    │
    │
    ml                   - modelling source code for use in this project.
    ├── __init__.py      - makes ml a Python module
    ├── eval.py          - evaluation script
    ├── train.py         - training script
    │
    ├── data             - scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features         - scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    └── model            - model binaries and inference pipeline scripts
    │   └── pipeline.py  - orchestrate the whole pipeline
    │
    └── tests            - pytest
    │
    │
    ├── Dockerfile       - The Dockerfile to build the container image.
    ├── lambda_dockerfile - The Dockerfile to build the container image for AWS Lambda.
    ├── Makefile         - Makefile for build and deployment commands.
    ├── requirements.txt - The pip requirements to be installed during the container build.
    └── template.yaml    - A template that defines the SAM application's AWS resources.

## GCP

Deploying inference service to Cloud Run

### Authenticate

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

### Authenticate

1. Install `awscli` and `sam-cli`
2. `aws configure`

### Deploy to Lambda

1. Run `sam build`
2. Run `sam deploy --guiChange this portion for other types of models

## Add the correct type hinting when completed

`aws cloudformation delete-stack --stack-name <STACK_NAME_ON_CREATION>`
