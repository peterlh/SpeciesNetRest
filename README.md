# SpeciesNet Service

A REST API service that uses Google's CameraTrapAI SpeciesNet model for species detection in camera trap images.

## Features

- FastAPI-based REST service with Swagger documentation
- Species detection using CameraTrapAI's SpeciesNet model
- Docker support with GPU acceleration
- Simple POST endpoint for image analysis

## API Endpoints

### POST /detect

Upload an image to get species detection results.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
    "success": true,
    "detections": [
        // Detection results from SpeciesNet model
    ]
}
```

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the service:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

## Running with Docker

The service is included in the main docker-compose.yml file. To run it:

```bash
docker-compose up speciesnet
```

## API Documentation

Once running, access the Swagger documentation at:
- http://localhost:8000/docs
- http://localhost:8000/redoc

## Requirements

- Python 3.10+
- NVIDIA GPU (recommended)
- CUDA support for TensorFlow
- Docker with NVIDIA Container Toolkit (for containerized deployment)
