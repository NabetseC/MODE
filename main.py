import mediapipe as mp
import cv2 as cv



def remove_edges(video_path, start, end, outpath):

    out_time = 0
    frames = 0

    cap = cv.VideoCapture(video_path)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = round(cap.get(cv.CAP_PROP_FPS))
    
    print("fps: ", fps)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(outpath, fourcc, fps, (frame_width, frame_height))

    saving = False

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

        if out_time == start:
            saving = True
        if out_time == end:
            saving = False
        
        
    cap.release()
    out.release()


def split_video(video_path, out_path, cut_size, time_between, start, end):
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
    fps = round(cap.get(cv.CAP_PROP_FPS))

    new_fps = 30

    if fps<30:

        print("Yo gng, it needs to be more than or equal to 30 fps")
        raise ValueError 
    
    print("fps: ", fps)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    saving = True

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames += 1
        
        if saving and time >= start:
            out.write(frame)
    
        if (frames == fps):
            frames = 0 
            out_time += 1 #second has passed
            time += 1

        if time == start:
            frame = 0
            out_time = 0
            saving = True


        if time == end:
            break
        
        elif not saving and out_time == time_between:
            saving = True
            out_time = 0

        elif out_time == cut_size:
            if time_between != 0:
                saving = False
            out_time = 0
        
    cap.release()
    out.release()


def human_classification(video_path,out_path):
    """
    given an input video, have human classify what type of punch it was
    features: 1,2,3,4,5,6 (account for body shot/head shot variation?)
    for now keep it only to those six, ignore body/head variation

    we also ignore if its good or bad, we just want to say what type of punch it was.
    Could later add more choices for what type of mistake it was
    """


    print("loading")

#remove_edges("./videos/canelo1.mp4", 277, 3033, "./videos/caneloOut.mp4")
split_video("./videos/canelo1.mp4","./videos/cropped_vids/canelo1.mp4",360,60, 277, 3033)

        
        
