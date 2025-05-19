# {{cookiecutter.project_name}}

{{cookiecutter.project_short_description}}

## For dev

### Requirements

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

## Project structure

Files related to application are in the `app` or `tests` directories.
Application parts are:

```text
├── app
│    ├── __init__.py          - makes app a Python module
│    │
│    │ # Fast-API stuff
│    ├── api                  - web related stuff.
│    │   └── routes           - web routes.
│    ├── core                 - application configuration, startup events, logging.
│    ├── models               - pydantic data schemas.
│    ├── services             - processes/logic that are not part of the ML modelling.
│    ├── main-aws-lambda.py   - [Optional] FastAPI application for AWS Lambda creation and configuration.
│    └── main.py              - FastAPI application creation and configuration.
│
│ # ML stuff
├── data            - where you persist data locally
│   ├── interim     - intermediate data that has been transformed.
│   ├── processed   - the final, canonical data sets for modeling.
│   └── raw         - the original, immutable data dump.
│
├── notebooks       - Jupyter notebooks. Naming convention is a number (for ordering),
│
├── ml              - modelling source code for use in this project.
│   ├── __init__.py - makes ml a Python module
│   │
│   ├── data        - scripts to download or generate data using auxiliary tables
│   │   └── make_dataset.py
│   │
│   ├── features    - scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   └── model                 - scripts to train models and make predictions
│   │   ├── model_runner.py   - wrapper classes for modular usage of ML models
│   │   ├── pipeline.py       - scripts to orchestrate the whole pipeline
│   │   └── result_runner.py  - pre and post processing of ML related I/O
│   │
│   ├── eval.py
│   └── train.py
│
└── tests           - pytest
```

## For prod

- clone the repo
- go to the project folder; run cmd 'docker build'
- upload the docker image to AWS and the env is ready

- how to run the code and see what it does!

### Installation

```sh
python -m venv venv
source venv/bin/activate
pip install poetry
make install
```

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

Made by <https://github.com/arthurhenrique/cookiecutter-fastapi/graphs/contributors> with ❤️

### Runnning Localhost

`make run`

## Access Swagger Documentation

> <http://localhost:8080/docs>

## Access Redocs Documentation

> <http://localhost:8080/redoc>

Made by <https://github.com/viniciusnvcosta/ai-dat/graphs/contributors> with ❤️
