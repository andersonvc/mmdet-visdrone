FROM python:3.9-slim-buster

ENV BACKEND_STORE_URI="/app/mlruns"
ENV MLFLOW_UI_PORT=5000
ENV MLFLOW_WORKERS=2

RUN pip3 install mlflow psycopg2-binary

WORKDIR /app

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri postgresql://mlflow:mlflow@host.docker.internal:5432/mlflow \
    --default-artifact-root ${BACKEND_STORE_URI} \
    --host 0.0.0.0 \
    --port ${MLFLOW_UI_PORT} \
    --workers ${MLFLOW_WORKERS}