FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model on first startup if artifacts are missing
CMD ["sh", "-c", \
     "python -m src.train && uvicorn src.service:app --host 0.0.0.0 --port 8000"]

EXPOSE 8000
