# Use an official lightweight Python image.
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (more robustly for Cloud/Render)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install via pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend files
COPY . .

# Expose the API port
EXPOSE 8000

# Set default env vars for production
# Render/Cloud usually provides these, but we set defaults here.
ENV DOCUMENTS_DIR=/app/documents
ENV CHROMA_PERSIST_DIR=/app/chroma_db

# Create volumes for persistent data
RUN mkdir -p /app/documents /app/chroma_db

# Command to run the backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
