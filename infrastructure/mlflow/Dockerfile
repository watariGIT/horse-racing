FROM python:3.10-slim

RUN pip install --no-cache-dir "mlflow>=2.10.0,<2.20" "google-cloud-storage>=2.14.0,<3"

RUN useradd -r -s /bin/false mlflow
USER mlflow

ENV PORT=8080
ENV MLFLOW_BACKEND_STORE_URI=""

EXPOSE ${PORT}

CMD exec mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --host 0.0.0.0 \
    --port ${PORT}
