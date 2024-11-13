import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
import io
import osmnx as ox
from shapely.geometry import Polygon, LineString, Point
import scipy
import re

import requests
import json
from datetime import datetime
from PIL import Image, UnidentifiedImageError
import io

def get_responses(location: list, api_key: str, heading=False) -> list:
    """
    Retrieves a Google Street View image and capture date for a given location as metadata.

    Args:
        location (list): Latitude and longitude as [lat, lon].
        api_key (str): Google API key for authentication.
        heading (bool): Defines whether multiple directions of view are considered. Default is False.

    Returns:
        list or dict: If heading=True, returns a list of dictionaries with image data for each heading.
                      If heading=False, returns a single dictionary with image data.
    """
    
    heading_param = [90, 180, 270, 360]
    results = []  
    
    # Fetch metadata to get pano_id, date, and location information
    metadata_url = f'https://maps.googleapis.com/maps/api/streetview/metadata?location={location}&key={api_key}'
    metadata_response = requests.get(metadata_url)
    metadata = json.loads(metadata_response.text)

    pano_id = metadata.get('pano_id')
    date_str = metadata.get('date', '')
    location = metadata.get('location', location)  # Default to location if location is missing

    date = datetime.strptime(date_str, "%Y-%m") if date_str else None

    if heading:
        for i in heading_param:
            if pano_id:
                image_url = f'https://maps.googleapis.com/maps/api/streetview?size=400x400&fov=120&heading={i}&pitch=0&pano={pano_id}&key={api_key}'
            else:
                image_url = f'https://maps.googleapis.com/maps/api/streetview?size=400x400&location={location}&fov=120&heading={i}&pitch=0&key={api_key}'

            image_response = requests.get(image_url)
            try:
                image = Image.open(io.BytesIO(image_response.content))
            except UnidentifiedImageError:
                print(f"Error: Image not available for location {location} with heading {i}")
                image = None

            results.append({
                "image": image,
                "date": date,
                "location": location,
                "pano_id": pano_id
            })

        return results 
    if pano_id:
        image_url = f'https://maps.googleapis.com/maps/api/streetview?size=400x400&fov=120&pitch=0&pano={pano_id}&key={api_key}'
    else:
        image_url = f'https://maps.googleapis.com/maps/api/streetview?size=400x400&location={location}&fov=120&pitch=0&key={api_key}'

    image_response = requests.get(image_url)
    try:
        image = Image.open(io.BytesIO(image_response.content))
    except UnidentifiedImageError:
        print(f"Error: Image not available for location {location}")
        image = None

    return {
        "image": image,
        "date": date,
        "location": location,
        "pano_id": pano_id
    }

    
def greenviewindex(images: pd.Series) -> list:
    """
    Calculates the vegetation percentage for a series of images using a segmentation model.

    Args:
        images (pd.Series): A pandas Series of PIL Image objects.

    Returns:
        list: A list of vegetation percentages for each image (0-100 scale).
              Returns None if the image processing fails.
    """
    
    semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
    gvi_list = []
    
    for image in images:
        try:
            results = semantic_segmentation(image)
            vegetation = [result for result in results if result['label'] == 'vegetation']
            
            if vegetation:
                vegetation_mask = np.array(vegetation[0]['mask'])
                vegetation_pixels = np.count_nonzero(vegetation_mask)
                total_pixels = vegetation_mask.size
                vegetation_percentage = (vegetation_pixels / total_pixels) * 100
            else:
                vegetation_percentage = 0  
                
        except (UnidentifiedImageError, ValueError):
            vegetation_percentage = None
        
        gvi_list.append(vegetation_percentage)
    
    return gvi_list
   


    