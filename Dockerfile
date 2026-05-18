# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the Sentence-Transformers reranking model during Docker Build.
# This ensures weights (~2.2 GB) are baked into the image, preventing multi-minute 
# startup latencies or AWS load balancer health-check timeouts on EC2/ECS boot.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-large')"

# Copy the application code and necessary folders
# (We exclude frontend/ and node_modules/ via .dockerignore)
COPY . .

# Expose the port that the app runs on
EXPOSE 9000

# Command to run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 9000"]
