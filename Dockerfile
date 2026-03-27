FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r api/requirements.txt

COPY . /app

# Dataset download + train (needs network during docker build)
RUN python scripts/bootstrap_deploy.py

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["python", "/app/docker_entrypoint.py"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
