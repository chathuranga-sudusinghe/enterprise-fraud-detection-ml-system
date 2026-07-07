FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y gcc g++ libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api /app/api
COPY ml /app/ml
COPY artifacts/metrics /app/artifacts/metrics

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
