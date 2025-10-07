FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Add build args
ARG MODEL_DOWNLOAD_URL
ARG MD_DOWNLOAD_URL

# Copy download script and env file
COPY download_model.sh ./ 
RUN chmod +x download_model.sh

# Download model files
RUN MODEL_DOWNLOAD_URL=$MODEL_DOWNLOAD_URL MD_DOWNLOAD_URL=$MD_DOWNLOAD_URL ./download_model.sh

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
