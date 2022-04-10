"""
Used to rename files for consistency
"""
import os
from os import path


#path to folder
path='5-FemaleGlasses-Yawning'
files=os.listdir(path)
for file in files:
        print(os.path.splitext(file)[1])
        #append prefix to each file name and save
        os.rename(os.path.join(path,file),'FGYYY'+str(file))
    