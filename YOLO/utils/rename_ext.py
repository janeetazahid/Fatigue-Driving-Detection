"""
Used convert files from JPEG to PNG for consistency 
"""
from PIL import Image 
import os 

directory = r'.' 
for filename in os.listdir(directory): 
    if filename.endswith(".jpg"): 
        prefix = filename.split(".jpg")[0]
        im = Image.open(filename)
        im.save(prefix+'.png')  
    else: 
        continue
