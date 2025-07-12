FROM python:3.11-slim

Set environment variables

ENV PYTHONDONTWRITEBYTECODE 1 ENV PYTHONUNBUFFERED 1 ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}

Install system dependencies including poppler-utils and tesseract-ocr

RUN apt-get update && 
apt-get install -y --no-install-recommends 
poppler-utils 
tesseract-ocr 
&& rm -rf /var/lib/apt/lists/*

Set working directory

WORKDIR /app

Copy requirements first to leverage Docker cache

COPY requirements.txt .

Install Python dependencies

RUN pip install --no-cache-dir -r requirements.txt

Copy application code

COPY . .

Command to run the application

CMD ["python", "chatbot_demo.py"]