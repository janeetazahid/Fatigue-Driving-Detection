"""
Used to extract frames from video
"""
import os
import numpy as np
import cv2
from glob import glob

def create_dir(path):
    """
    Creates directory 
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")

def save_frame(video_path, save_dir, gap=10):
    """
    Saves frames from video 
    @param video_path: path to video 
    @param save_dir: where frames should we saved
    @param gap: how often to save image of frame
    """
    #get name of video 
    name = video_path.split("/")[-1].split(".")[0]
    #create save path
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)
    #start video capture 
    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        #capsture frame 
        ret, frame = cap.read()
        #vidoe over 
        if ret == False:
            cap.release()
            break
        #always save first frame 
        if idx == 0:
            cv2.imwrite(f"{save_path}/{idx}.png", frame)
        else:
            #else if gap value reached save frame 
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{idx}.png", frame)
        idx += 1
        if idx==5000:
            break

if __name__ == "__main__":
    #enter video path here
    video_paths = glob("0.mp4")
    #directory where video is saved
    save_dir = "test"

    for path in video_paths:
        save_frame(path, save_dir, gap=1)

