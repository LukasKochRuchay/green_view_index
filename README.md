# Green View Index to measure Perceived Bikeability
==============================

A green view index for perceived bikeability based on Google Street View API and open geodata of the city of Berlin using image segmentation. When it comes to bikeability, it’s about more than just having bike lanes. Perceptual factors like greenery and overall street aesthetics play a crucial role in how comfortable and enjoyable a street feels for cyclists. Enhancing green infrastructure isn’t only about climate resilience and reducing urban heat—it’s also a chance to improve the experience of biking and walking, making active mobility more appealing and accessible for everyone.

### Project Organization


├── .devcontainer  
│  
├── notebooks  
│   ├── 01_get_street_view_images.ipynb  
│   └── 02_calculate_greenview.ipynb  
│  
├── src  
│   ├── __init__.py  
│   └── gvi_funcs.py  
│  
├── .env  
├── .gitattributes  
├── .gitignore  
├── Dockerfile  
└── README.md  

=======


Example Output
Below is an example of the images used in this project:

<p align="center">  
  <img src="ressources/pic1.png" alt="Google Street View Image" width="40%" style="margin-right: 20px;" />  
  <img src="ressources/pic2.png" alt="Vegetation Segmentation" width="40%" style="margin-left: 20px;" />  
</p>

Left: Google Street View Image
Right: Vegetation Segmentation

Additional Information
Data: Street View Static API overview
Model: NVIDIA SegFormer-B5 (fine-tuned on Cityscapes 1024x1024)