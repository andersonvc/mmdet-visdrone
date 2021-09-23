FROM python:3.9-slim-buster

ENV BACKEND_STORE_URI="/data"
ENV MLFLOW_UI_PORT=5000
ENV MLFLOW_WORKERS=2

RUN pip3 install mlflow

WORKDIR /data

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${BACKEND_STORE_URI} \
    --host 0.0.0.0 \
    --port ${MLFLOW_UI_PORT} \
    --workers ${MLFLOW_WORKERS}