import mediapipe as mp
import cv2 as cv



def split_video(video_path, out_path, cut_size, time_between):
    #ideally pass in a video with fps 30

    """
    This works for time_between = 0, whihc is useless since it means the
    video is unchanged

    Given a video, splits it into x cut_size second sections and combines it
    into one output
    adjustable time between cuts
    """

    print("wait")
    time = 0
    out_time = 0
    frames = 0

    cap = cv.VideoCapture(video_path)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    saving = True

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames += 1
        if saving:
            out.write(frame)
    
        if (frames == fps):
            frames = 0 
            out_time += 1 #second has passed
        
        if not saving and out_time == time_between:
            saving = True
            out_time = 0

        elif out_time == cut_size:
            if time_between != 0:
                saving = False
            out_time = 0
    cap.release()
    out.release()

split_video("./videos/OPM.mp4","./videos/cropped_vids/OPM.mp4",10,10)

        
        
