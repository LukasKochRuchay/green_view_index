import numpy as np
import pandas as pd
import requests
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
import io


def fetch_image(image_url: str) -> Image.Image:
    """
    Helper function to fetch and open the image.
    """
    try:
        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))
        return image
    except UnidentifiedImageError:
        print(f"Error: Image not available for URL {image_url}")
        return None


def get_data(location: list, api_key: str, heading=False) -> list:
    """
    Retrieves a Google Street View image and capture metadata, that is, date, location and pano_id.

    Args:
        location (list): Latitude and longitude as [lat, lon].
        api_key (str): Google API key for authentication.
        heading (bool): Defines whether multiple directions of view are considered. Default is False.

    Returns:
        list : returns a list of dictionaries with image data for each heading.
    """
    
    heading_param = [90, 180, 270, 360] if heading else [0]
    results = [] 
        
    metadata = requests.get(f'https://maps.googleapis.com/maps/api/streetview/metadata?location={location}&key={api_key}').json()
    
    pano_id = metadata.get('pano_id')
    date_str = metadata.get('date', '')
    location = metadata.get('location', '')
        
    for i in heading_param:
        if pano_id:
            image_url = f'https://maps.googleapis.com/maps/api/streetview?size=400x400&fov=120&heading={i}&pitch=0&pano={pano_id}&key={api_key}'
        else:
            image_url = f'https://maps.googleapis.com/maps/api/streetview?size=400x400&location={location}&fov=120&heading={i}&pitch=0&key={api_key}'

        image = fetch_image(image_url)
     
        results.append({
            "image": image,
            "date": datetime.strptime(date_str, "%Y-%m") if date_str else None,
            "location": location,
            "pano_id": pano_id
    })
    
    return results
    

    
def segmentation_index(images: pd.Series, label='vegetation') -> list:
    """
    Calculates the vegetation percentage for a series of images using a segmentation model.

    Args:
        images (pd.Series): A pandas Series of PIL Image objects.
        label (str): A string of possibles lables of semantic_segmentation_pipelin.
        That is: road, sidewalk, building, wall, pole, traffic sign, vegetation, terrain, sky, car,person.
        By default label is vegeation. 

    Returns:
        list: A list of percentages of a segmentation mask for each image (0-100 scale). E. g. percentage of pixels that are classified as label when label == vegetatrion
              Returns None if the image processing fails.
    """
    
    semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
    segmentation_list = []
    
    for image in images:
        try:
            results = semantic_segmentation(image)
            label_result = [result for result in results if result['label'] == label]
            
            if label_result:
                mask = np.array(label_result[0]['mask'])
                pixels = np.count_nonzero(mask)
                total_pixels = mask.size
                percentage = (pixels / total_pixels) * 100
            else:
                percentage = 0  
                
        except (UnidentifiedImageError, ValueError):
            percentage = None
        
        segmentation_list.append(percentage)
    
    return segmentation_list
   


    