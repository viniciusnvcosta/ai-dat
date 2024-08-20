FROM python:3.11

ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    mv /root/.local/bin/poetry /usr/local/bin/ && \
    pip install --upgrade pip && \
    poetry config virtualenvs.create false

# Run Poetry
COPY poetry.lock pyproject.toml ./

ARG DEV=false
RUN if [ "$DEV" = "true" ] ; then poetry install --with dev --sync ; else poetry install --no-interaction --no-ansi --sync ; fi

COPY ./app/ ./
COPY ./ml/model/ ./ml/model/

ENV PYTHONPATH "${PYTHONPATH}:/app"

EXPOSE 8080
# -------- replace with comment to run with gunicorn --------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
# CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker". "-b", "0.0.0.0:8080"]