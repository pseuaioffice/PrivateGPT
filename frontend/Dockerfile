# Base Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Expose flask port
EXPOSE 5000

# Default env vars for backend endpoint
# In cloud, this will be set to the backend internal service URL.
ENV BACKEND_URL=http://localhost:8000
ENV HOST=0.0.0.0
ENV PORT=5000
ENV DEBUG=False

# Command
CMD ["python", "app.py"]
