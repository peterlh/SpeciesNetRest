from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import json
import os
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms
import io
from PIL import Image
import yolov5

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure logging
logging.basicConfig(level=logging.INFO)
# Reduce verbosity of detailed processing logs
logging.getLogger().setLevel(logging.INFO)

app = FastAPI(
    title="SpeciesNet Service",
    description="REST API for species detection using CameraTrapAI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and metadata
speciesnet_model = None
megadetector_model = None
species_metadata = None
MODEL_PATH = '/app/model'

# Load MegaDetector settings from environment
MD_CONFIDENCE_THRESHOLD = float(os.getenv('MD_CONFIDENCE_THRESHOLD', '0.2'))

# MegaDetector class mapping
MD_CLASSES = {
    0: 'animal',
    1: 'person',
    2: 'vehicle'
}

def find_file_by_pattern(directory, pattern):
    """Find a file in directory that matches the given pattern"""
    matches = [f for f in os.listdir(directory) if pattern in f.lower()]
    if not matches:
        raise FileNotFoundError(f"No file matching pattern '{pattern}' found in {directory}")
    return os.path.join(directory, matches[0])

@app.on_event("startup")
async def startup_event():
    """Initialize models and metadata on startup"""
    global speciesnet_model, megadetector_model, species_metadata
    try:
        # Find and load SpeciesNet model
        model_file = find_file_by_pattern(MODEL_PATH, '.keras')
        logging.info(f"Found SpeciesNet model: {os.path.basename(model_file)}")
        speciesnet_model = tf.keras.models.load_model(model_file)
        
        # Load MegaDetector model
        md_file = find_file_by_pattern(MODEL_PATH, '.pt')
        logging.info(f"Found MegaDetector model: {os.path.basename(md_file)}")
        megadetector_model = yolov5.load(md_file)
        megadetector_model.eval()
        
        # Find geofence file (*.json)
        geofence_file = find_file_by_pattern(MODEL_PATH, 'geofence')
        logging.info(f"Found geofence file: {os.path.basename(geofence_file)}")
        with open(geofence_file, 'r') as f:
            species_metadata = json.load(f)
        
        logging.info("Service initialized successfully")
            
    except Exception as e:
        logging.error(f"Failed to initialize model: {str(e)}")
        raise

def filter_species_by_country(predictions, country_code):
    """
    Filter species predictions based on country code and geofence rules.
    
    Args:
        predictions: Model predictions array
        country_code: ISO country code (e.g., 'DNK' for Denmark)
        
    Returns:
        List of filtered species predictions that are valid for the given country
    """
    filtered_results = []
    
    # Find and load species labels file
    labels_file = find_file_by_pattern(MODEL_PATH, 'labels')
    with open(labels_file, 'r') as f:
        species_labels = [line.strip() for line in f.readlines()]
    
    logging.info(f"Number of species labels: {len(species_labels)}")
    
    # Parse labels into tuples with taxonomy information
    parsed_labels = []
    for label in species_labels:
        parts = label.split(';')
        if len(parts) >= 2:
            uuid = parts[0]
            taxa = [p for p in parts[1:] if p]  # Remove empty strings
            
            # Build taxonomy paths for hierarchical lookup
            taxa_paths = []
            current_path = []
            for taxon in taxa:
                current_path.append(taxon)
                taxa_paths.append(';;;;'.join(current_path))
            
            common_name = taxa[-1] if taxa else 'Unknown'
            class_name = taxa[0] if taxa else 'Unknown'
            parsed_labels.append((uuid, class_name, common_name, taxa_paths))
    
    # Process predictions
    pred_list = [(i, float(conf)) for i, conf in enumerate(predictions[0])]
    pred_list.sort(key=lambda x: x[1], reverse=True)
    
    # Log top predictions before filtering
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug("Top 10 predictions before filtering:")
        for species_id, confidence in pred_list[:10]:
            if species_id < len(parsed_labels):
                uuid, class_name, common_name, taxa_paths = parsed_labels[species_id]
                logging.debug(f"  {class_name} - {common_name}: {confidence:.3f}")
    
    # Filter predictions
    for species_id, confidence in pred_list[:10]:
        if confidence > 0.1 and species_id < len(parsed_labels):
            uuid, class_name, common_name, taxa_paths = parsed_labels[species_id]
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Checking species {common_name} for country {country_code}")
            
            # Check each taxonomic level
            is_allowed = False
            most_specific_level = None
            
            # Check from most specific to least specific taxa level
            for taxa_path in reversed(taxa_paths):
                # Add trailing ;;;; to match the metadata format
                lookup_path = taxa_path + ';;;;'
                
                if lookup_path in species_metadata:
                    metadata = species_metadata[lookup_path]
                    allowed_countries = metadata.get('allow', {})
                    
                    if country_code in allowed_countries:
                        is_allowed = True
                        most_specific_level = taxa_path
                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            logging.debug(f"Species {common_name} allowed in {country_code} at {taxa_path}")
                        break
            
            if is_allowed:
                filtered_results.append({
                    'species_id': species_id,
                    'uuid': uuid,
                    'class_name': class_name,
                    'common_name': common_name,
                    'confidence': confidence,
                    'taxa_level': most_specific_level
                })
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Added {common_name} to results (allowed at {most_specific_level})")
    
    return filtered_results

def run_megadetector(image):
    """Run MegaDetector on image to detect animals"""
    # Run inference
    results = megadetector_model(image)
    logging.info(f"MegaDetector results: {results}")
    logging.info(f"Predictions: {results.pred}")
    
    # Process results
    animal_detected = False
    if len(results.pred[0]) > 0:
        # Each detection is [x1, y1, x2, y2, conf, cls]
        for detection in results.pred[0]:
            class_id = int(detection[-1].item())  # Last element is class
            confidence = float(detection[-2].item())  # Second to last is confidence
            logging.info(f"Detection - class: {class_id}, confidence: {confidence}")
            if class_id == 0 and confidence >= MD_CONFIDENCE_THRESHOLD:  # 0 is 'animal'
                animal_detected = True
                break
    
    return animal_detected

@app.post("/detect", tags=["Detection"])
async def detect(
    file: UploadFile = File(description="Image file to analyze"),
    country: str = Form(description="ISO country code (e.g., 'DK' for Denmark)")
):
    """Detect species in image and filter by country"""
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # First run MegaDetector
        animal_detected = run_megadetector(image)
        
        if not animal_detected:
            return {
                "success": True,
                "message": "No animals detected in image",
                "country": country,
                "detections": []
            }
        
        # If animal detected, run SpeciesNet
        # Resize to match model input size
        speciesnet_image = image.resize((480, 480))
        
        # Convert to numpy array and add batch dimension
        img_array = np.array(speciesnet_image)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        # Run prediction
        predictions = speciesnet_model.predict(img_array)
        
        # Filter predictions by country
        country_mapping = {
            'DK': 'DNK',
            'GB': 'GBR',
            'US': 'USA',
            'SE': 'SWE',
            'NO': 'NOR',
            'FI': 'FIN',
            'DE': 'DEU',
            'NL': 'NLD',
            'BE': 'BEL',
            'FR': 'FRA'
        }
        iso3_country = country_mapping.get(country.upper(), country.upper())
        filtered_predictions = filter_species_by_country(predictions, iso3_country)
        
        return {
            "success": True,
            "country": country,
            "country_iso3": iso3_country,
            "detections": filtered_predictions
        }
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
