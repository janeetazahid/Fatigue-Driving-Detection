# -*- coding: utf-8 -*-
"""
Code used to train the yolo model. Training was done on Google Colab
"""

#imports 
import subprocess as sbp
import os
import re
# %tensorflow_version 1.x
import tensorflow as tf
tf.__version__

#clone the darknet directory from github which is used for training 
!git clone https://github.com/AlexeyAB/darknet.git #command only works on Google Colab 

def imShow(path):
  """
  Displays image specified by path 
  @param path: path to image
  """
  import cv2
  import matplotlib.pyplot as plt
#   %matplotlib inline
  #read image
  image = cv2.imread(path)
  #extract height and width
  height, width = image.shape[:2]
  #resize
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
  #plot image
  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

#unzip the obj folder, this folder contains the dataset
!unzip obj.zip #command only works on Google Colab 

# make nessacary changes to the configuration file
# set the number of max_batches (2000*number of classes)
max_batch=10000
# calculate 2 steps values
step1 = 0.8 * max_batch
step2 = 0.9 * max_batch

#adjust the number of classes and filter size 
num_classes=5
num_filters = (num_classes + 5) * 3

#path to configuration file 
cfg_file = '/content/yolov4-tiny_obj.cfg'
#update configuration file 
with open(cfg_file) as f:
    s = f.read()
s = re.sub('max_batches = \d*','max_batches = '+str(max_batch),s)
s = re.sub('steps=\d*,\d*','steps='+"{:.0f}".format(step1)+','+"{:.0f}".format(step2),s)
s = re.sub('classes=\d*','classes='+str(num_classes),s)
s = re.sub('pad=1\nfilters=\d*','pad=1\nfilters='+"{:.0f}".format(num_filters),s)

with open(cfg_file, 'w') as f:
  f.write(s)

#change the makefile to use OPENCV and the GPU (speeds up the porcess)
# %cd /content/darknet
!sed -i 's/OPENCV=0/OPENCV=1/g' Makefile #command only works on Google Colab 
!sed -i 's/GPU=0/GPU=1/g' Makefile #command only works on Google Colab 
!make #command only works on Google Colab 
!chmod +x ./darknet #command only works on Google Colab 

#train the model
# %cd /content/darknet/
!./darknet detector train /content/obj.data /content/yolov4-tiny_obj.cfg /content/yolov4-tiny.conv.29 -dont_show -ext_output -map #command only works on Google Colab 

#test the model 
!./darknet detector map /content/obj.data /content/yolov4-tiny_obj.cfg "/content/darknet/backup/yolov4-tiny_obj_best.weights" -points 0 #command only works on Google Colab 

#test the results of one image and show the output
!./darknet detector test /content/obj.data  /content/yolov4-tiny_obj.cfg  "/content/darknet/backup/yolov4-tiny_obj_best.weights" /content/closed_eye2.jpg -ext_output #command only works on Google Colab 
imShow('predictions.jpg')

#test the results of on video 
!./darknet detector demo /content/obj.data /content/yolov4-tiny_obj.cfg "/content/yolov4-tiny_obj_best.weights" /content/vid.avi -out_filename /content/results2.mp4 -dont_show #command only works on Google Colab 

