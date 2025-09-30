#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

# Check if MODEL_DOWNLOAD_URL is set
if [ -z "$MODEL_DOWNLOAD_URL" ]; then
  echo "Error: MODEL_DOWNLOAD_URL not set in .env file"
  exit 1
fi

# Create model directory
mkdir -p model

# Download SpeciesNet model files
echo "Downloading SpeciesNet model..."
curl -L -o model/model.tar.gz \
  "$MODEL_DOWNLOAD_URL"

# Extract the model files
echo "Extracting SpeciesNet model..."
tar -xzf model/model.tar.gz -C model/

# Clean up
rm model/model.tar.gz

# Download MegaDetector model
echo "Downloading MegaDetector model..."
curl -L -o model/md_v5a.0.0.pt \
  "${MD_DOWNLOAD_URL}"

echo "Model files downloaded and extracted successfully"
