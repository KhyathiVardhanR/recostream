FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY recostream recostream
COPY scripts scripts
RUN mkdir -p /tmp/prometheus
EXPOSE 8000 9300
# API by default; the same image runs the consumer, trainer and bootstrap jobs.
CMD ["uvicorn", "recostream.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
