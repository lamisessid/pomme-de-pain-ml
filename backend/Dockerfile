FROM python:3.11-slim

WORKDIR /app/backend
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY preprocessing/data/raw /app/preprocessing/data/raw

EXPOSE 8642
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8642"]
