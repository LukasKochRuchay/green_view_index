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


def get_responses(latlon: list, point: list, api_key: str) -> list:
    """
    Retrieves a Google Street View image and capture date for a given location as metadata.

    Args:
        latlon (list): Latitude and longitude as [lat, lon].
        point (list): Identifier or metadata for the location.
        api_key (str): Google API key for authentication.
        heading (bool): Derfines wheather a single direction of view is considered or not. Default is False

    Returns:
        list: [PIL.Image (image), datetime or None (date), list (point)]
    """
    
def get_responses(latlon: list, point: list, api_key: str, heading=False) -> list:
    heading_param = [90, 180, 270, 360]
    results = []  
    
    metadata_url = f'https://maps.googleapis.com/maps/api/streetview/metadata?location={latlon}&key={api_key}'
    metadata_response = requests.get(metadata_url)
    metadata = json.loads(metadata_response.text)

    pano_id = metadata.get('pano_id')
    date_str = metadata.get('date', '')
    location = metadata.get('location', latlon) 

    
    date = datetime.strptime(date_str, "%Y-%m") if date_str else None

    if heading:
             
        for i in heading_param:
            if pano_id:
                image_url = f'https://maps.googleapis.com/maps/api/streetview?size=400x400&fov=120&heading={i}&pitch=0&pano={pano_id}&key={api_key}'
            else:
                image_url = f'https://maps.googleapis.com/maps/api/streetview?size=400x400&location={latlon}&fov=120&heading={i}&pitch=0&key={api_key}'
            
            image_response = requests.get(image_url)
            image = Image.open(io.BytesIO(image_response.content))

            results.append({
                "image": image,
                "date": date,
                "location": location,
                "pano_id": pano_id,
                "point": point
            })

        return results  

    else:
        if pano_id:
            image_url = f'https://maps.googleapis.com/maps/api/streetview?size=400x400&fov=120&pitch=0&pano={pano_id}&key={api_key}'
        else:
            image_url = f'https://maps.googleapis.com/maps/api/streetview?size=400x400&location={latlon}&fov=120&pitch=0&key={api_key}'

        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))

        return {
            "image": image,
            "date": date,
            "location": location,
            "pano_id": pano_id,
            "point": point
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



def get_edges(G, lon: pd.Series, lat: pd.Series) -> list:
    """
    Finds unique nearest edges in a graph for given longitude and latitude points.

    Args:
        lon (pd.Series): A pandas Series of longitude coordinates.
        lat (pd.Series): A pandas Series of latitude coordinates.
        G (): a networkx.classes.multidigraph.MultiDiGraph

    Returns:
        list: A list of unique edges represented as tuples of node pairs (e.g., [(node1, node2), ...]).
              Returns an empty list if an error occurs.
    """
    try:
        edges = ox.nearest_edges(G, [lon], [lat])
        edges = list(dict.fromkeys(edges))  
        return [k[0:2] for k in edges]
    except Exception as e:
        return []

   
    
def extract_edge_data(edge_list: list, G):
    """
    Extracts geometry and name data from the first edge in a list of edges within a graph.

    Args:
        edge_list (list): A list of edges, where each edge is a tuple (u, v) of node IDs.
        G: A NetworkX graph or similar graph object with edge data.

    Returns:
        tuple: A tuple (geometry, name) where:
            - geometry (LineString or None): The geometry of the edge if available and of type LineString; otherwise None.
            - name (str or None): The name of the edge if available; otherwise None.
            Returns (None, None) if edge_list is empty or if the edge has no data.
    """
    if not edge_list:
        return None, None
    
    u, v = edge_list[0]
    edge_data = G.get_edge_data(u, v)
    
    if edge_data is not None:
        data = edge_data.get(0, {})
        geometry = data.get('geometry', None)
        name = data.get('name', None)
        
        if not isinstance(geometry, LineString):
            geometry = None
        
        return geometry, name
    else:
        return None, None

    