# Fatigue-Driving-Detection
Computer Engineering Capstone Project. <br>
This project aims to predict the level of fatigue of a driver using deep learning. <br>
The eyes and mouth of the driver are used as features and thier state (open or closed) is detected by a YOLO model. <br>
The count of how long the eyes are closed and mouth is open is fed into an LSTM model which outputs the level of the drivers fatigue (low, meduim, or high) <br>
The program also detects when the eyes are closed for a long period and when the driver is yawning using two LSTM models <br>



[LSTM](LSTM): Code and files associated with LSTM model <br>
[YOLO](YOLO): Code and files associated with YOLO model <br>
[config](config): Configuration files for YOLO model <br>
[model_dense300](model_dense300): Saved LSTM model that predicts fatigue <br>
[model_eye](model_eye): Saved LSTM model that predicts eye closure <br>
[model_yawn](model_yawn): Saved LSTM model that predicts yawns <br>
[fatigue_detect.py](fatigue_detect.py): Python code that detects fatigue, eye closure and yawns <br>
[result.avi](result.avi): Sample output result  <br>
[vid2.avi](vid2.avi): Sample video to perform detection on <br>



### Running Code
To run the code, run the fatigue_detect.py file. The file can be run with an addtional argument "-p path_to_video_file" or by default it will run on the webcam
