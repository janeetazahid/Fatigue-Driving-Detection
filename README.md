# Fatigue Driving Detection
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg) ![License](https://img.shields.io/badge/license-GNU-green.svg) ![YOLO](https://img.shields.io/badge/Model-YOLO-blue.svg) ![LSTM](https://img.shields.io/badge/Model-LSTM-blue.svg) ![Keras](https://img.shields.io/badge/Library-Keras-red.svg) ![CV2](https://img.shields.io/badge/Library-opencv-red.svg)<br>
This project aims to use biological indicators of fatigue (yawning and eye closure) along with computer vision to detect a drivers fatigue level. Fatigue driving causes many accidents on the road, however, the exsisting solutions to detect a drivers
fatigue are too costly. Our solution provides a cost-efficent solution that can warn the driver of fatigue before it is too late.  


## Features 

- The state of the driver's eye and mouth (open or closed) are detected using a YOLO model 
- LSTM model outputs the driver's fatigue level (low, meduim or high) using the results supplied by the YOLO model  
- The program also detects when the driver yawns and when her/his eyes are closed for an extended period (both which indicate fatigue) using LSTM models 

## Folder Breakdown 

[LSTM](LSTM): Code and files associated with LSTM model <br>
[YOLO](YOLO): Code and files associated with YOLO model <br>
[config](config): Configuration files for YOLO model <br>
[model_dense300](model_dense300): Saved LSTM model that predicts fatigue <br>
[model_eye](model_eye): Saved LSTM model that predicts eye closure <br>
[model_yawn](model_yawn): Saved LSTM model that predicts yawns <br>
[fatigue_detect.py](fatigue_detect.py): Python code that detects fatigue, eye closure and yawns <br>
[result.avi](result.avi): Sample output result  <br>
[vid2.avi](vid2.avi): Sample video to perform detection on <br>

## TechStack
- Keras
- Tensorflow
- CV2
- Numpy


## Running Code

**1.** Clone repo
```
$ git clone https://github.com/janeetazahid/Fatigue-Driving-Detection.git
```

**2.** Install the required libraries by running the following command
```
pip install -r requirements.txt
```

**3.** Begin the fatigue detection by running the following command 
```
python fatigue_detect.py -p "vid2.avi"
```

Any path for a video file can be supplied after the -p argument. If no path is supplied the webcam will be used 



