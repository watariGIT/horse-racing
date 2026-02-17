FROM python:3.10-slim

RUN pip install --no-cache-dir "mlflow==2.10.2" google-cloud-storage

ENV PORT=8080
ENV MLFLOW_BACKEND_STORE_URI=""

EXPOSE ${PORT}

CMD exec mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --serve-artifacts
