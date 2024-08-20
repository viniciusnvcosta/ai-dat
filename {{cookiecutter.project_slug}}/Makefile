SHELL := /bin/bash

# Import variables from .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Variables definitions
# -----------------------------------------------------------------------------

ifeq ($(TIMEOUT),)
TIMEOUT := 60
endif

ifeq ($(MODEL_PATH),)
MODEL_PATH := {{cookiecutter.machine_learn_model_path}}
endif

ifeq ($(MODEL_NAME),)
MODEL_NAME := {{cookiecutter.machine_learn_model_name}}
endif

ACCOUNT_ID = $(shell aws sts get-caller-identity --query "Account" --output text)
# ECR_URI = $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(ECR_REPO_NAME):$(IMAGE_TAG)

# Target section and Global definitions
# -----------------------------------------------------------------------------
.PHONY: help all install run test sync test compose down generate_dot_env

all: clean test sync install run deploy down

help:echo data
	@echo ""
	@echo "Targets:"
	@echo "  install           Install dependencies"
	@echo "  sync              Sync the application data to S3"
	@echo "  test              Run tests"
	@echo "  compose           Run docker-compose"
	@echo "  down              Stop docker-compose"
	@echo "  generate_dot_env  Generate .env file"
	@echo "  clean             Clean up"

install: generate_dot_env
	pip install --upgrade pip
	poetry install --with dev --sync

run:
	PYTHONPATH=app/ poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Sync the application data
.PHONY: sync
sync:
	@echo "Syncing the application data..."

test:
	poetry run pytest tests -vv --show-capture=all

compose: generate_dot_env
	docker-compose build
	docker-compose up -d

down:
	docker-compose down

generate_dot_env:
	@if [[ ! -e .env ]]; then \
		cp .env.example .env; \
	fi

# Clean up
.PHONY: clean
clean:
	@find . -name '*.pyc' -exec rm -rf {} \;
	@find . -name '__pycache__' -exec rm -rf {} \;
	@find . -name 'Thumbs.db' -exec rm -rf {} \;
	@find . -name '*~' -exec rm -rf {} \;
	rm -rf .aws-sam
	rm -rf .cache
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf htmlcov
	rm -rf .tox/
	rm -rf docs/_build