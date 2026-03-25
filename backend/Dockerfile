# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and model artifacts
COPY main.py .
COPY xgb_model.json .
COPY road_features_lookup.csv .
COPY model_metadata.json .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
