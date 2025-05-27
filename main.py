import mediapipe as mp
import cv2 as cv


#Given a video, splits it into x minute sections
#adjustable time between cuts
def split_video(video, cut_size, end, start = 0):
    print("wait")