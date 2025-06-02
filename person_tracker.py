import cv2
import threading

from ultralytics import YOLO

import numpy as np
import os

# Load the YOLO11 model
model = YOLO("yolo11n-pose.pt")

# Open the video file


#Helper functions:
txt_to_move = {0:"block", 1:"jab", 2:"straight", 3:"left_hook", 4:"right_hook", 5:"left_uppercut", 6:"right_uppercut"}
move_to_txt = {"block":0,"jab":1, "straight":2, "left_hook":3, "right_hook":4, "left_uppercut":5, "right_uppercut":6}

"""
dictionary listing the next index for data input at each folder location
helps avoid overwriting or manually adjusting things
"""
next_idx = {0:0,1: 0, 2: 0,3:0,4:0, 5:0, 6:0 }

for file in os.listdir("./data"):
    loc = "./data/"+file
    sd = os.listdir(loc)
    
    if len(sd) != 0:
        next_idx[move_to_txt[file]] = int(sd[-1][:-4])+1
    else:
        next_idx[move_to_txt[file]] = 0

print(next_idx)
        



def ang_calc(a,b,c):
    p1 = np.array(a)
    p2 = np.array(b)
    p3 = np.array(c)

    print("p1: ", p1)
    print("p2: ", p2)
    print("p3: ", p3)

    rads = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    angle = np.round(abs(rads * (180/np.pi)))

    if angle > 180:
        return 360 - angle 
    else:
        return angle 

def kpt_to_angles(pkt_points):

    #right hemi
    total = []
    for kpts in pkt_points:
        print("keypoints: ", kpts)
        right_wrist = [kpts[10][0], kpts[10][1]]
        right_elbow = [kpts[8][0], kpts[8][1]]
        right_shoulder = [kpts[6][0], kpts[6][1]]
            
        rightElbow_angle = ang_calc(right_wrist, right_elbow, right_shoulder)

        right_hip = [kpts[12][0], kpts[12][1]]
        rightShoulder_angle = ang_calc(right_elbow, right_shoulder, right_hip)

        right_knee = [kpts[14][0], kpts[14][1]]
        rightHip_angle = ang_calc(right_shoulder, right_hip, right_knee)

        right_ankle = [kpts[16][0], kpts[16][1]]
        rightKnee_angle = ang_calc(right_hip, right_knee, right_ankle)

        #left hemi
        left_wrist = [kpts[9][0], kpts[9][1]]
        left_elbow = [kpts[7][0], kpts[7][1]]
        left_shoulder = [kpts[5][0], kpts[5][1]]
            
        leftElbow_angle = ang_calc(left_wrist, left_elbow, left_shoulder)

        left_hip = [kpts[11][0], kpts[11][1]]
        leftShoulder_angle = ang_calc(left_elbow, left_shoulder, left_hip)

        left_knee = [kpts[13][0], kpts[13][1]]
        leftHip_angle = ang_calc(left_shoulder, left_hip, left_knee)

        left_ankle = [kpts[15][0], kpts[15][1]]
        leftKnee_angle = ang_calc(left_hip, left_knee, left_ankle)

        angles = [
            rightElbow_angle, 
            rightShoulder_angle, 
            rightHip_angle, 
            rightKnee_angle, 
            leftElbow_angle, 
            leftShoulder_angle,
            leftHip_angle,
            leftKnee_angle
            ]

        total.append(angles)
    return total

def get_user_input():
    global start
    global pending 
    global results_buffer 
    global cycles
    global frames 


    print("input length: ", len(results_buffer))

    starting_boxes = {}

    boxes = results_buffer[0][0].boxes

    for i in range(len(boxes.cls)):
        if int(boxes.cls[i]) == 0: #box of humam
            starting_boxes[int(boxes.id[i])] = []

    for frame in results_buffer:
        boxes = frame[0].boxes
        for person_id, person in enumerate(frame[0].keypoints):
                for i in range(len(boxes.cls)):
                    if boxes.id is not None and int(boxes.id[i]) in starting_boxes: #box of humam
                            keypoints = person.xy[0]
                            #angles = kpt_to_angles(keypoints[0]) could calculate angles for everyone, if angle can't be made then discard
                            left_hip = keypoints[11]
                            x, y = left_hip[0], left_hip[1]
                            xs, ys, xb, yb = boxes.xyxy[0][0], boxes.xyxy[0][1], boxes.xyxy[0][2], boxes.xyxy[0][3]
                            if x>= xs and x<=xb and y>= ys and y<= yb:
                                starting_boxes[int(boxes.id[i])].append(keypoints)
 

    for key in starting_boxes:
        if len(starting_boxes[key]) != 40:
            continue
        else:
            print("Salutations, please classify or discard boxer ", key )
            user_input = input("0:block | 1:jab | 2:straight | 3:lefthook | 4:righthook | 5:leftuppercut | 6:rightuppercut | else discard")
            print(f"You entered: {user_input}")
            """
                Here , depending on user input, save angles from the boxer to some local folder
            """
            if user_input.isdigit() and int(user_input) in txt_to_move:
                angles = kpt_to_angles(starting_boxes[key])
                np.save(f"./data/{txt_to_move[int(user_input)]}/{next_idx[int(user_input)]}.npy", angles)
                next_idx[int(user_input)] += 1
                print("saved")
            else:
                print("discarded")



    frames = frames - 40
    cycles += 1
    start +=  40
    pending = False
    results_buffer = []

# Loop through the video frames

#set_start if you want to skip to some point the fight when starting progrm
def play_video(video_path, set_start = 0):
    #set start is in seconds, however cap.set takes frame index
    #multiply seconds by fps(30)
    global start
    global pending 
    pending = False
    start = set_start*30
    cap = cv2.VideoCapture(video_path)
    global frames 
    frames= 0

    global cycles
    cycles = 0

    global results_buffer 
    results_buffer = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)


    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames

            results = model.track(frame, persist=True, conf = .6, verbose = False)
            annotated_frame = results[0].plot()

            for person_id, person in enumerate(results[0].keypoints):
            #print("boxes.cls[i]: ", int(boxes.cls[i]))
                if(person.conf is not None):
                    #print(person.xy)
                    kpt = person.xy.tolist()
                    #print("kpts: ", kpt)
                    kpt = kpt[0][13]
                    x, y = int(kpt[0]), int(kpt[1])
                    #print("person id: ", int(person_id))
                    cv2.putText(annotated_frame, str(int(person_id)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.rectangle(annotated_frame, (x-10,y-10), (x+10,y+10), (255, 0, 104), 2, -1)


            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)
            if not pending and frames >=0:
                print(frames)
                results_buffer.append(results)

            if frames % 39  == 0 and frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                if not pending:
                    pending = True
                    input_thread = threading.Thread(
                    target=get_user_input,
                    args=()
                    )
                    input_thread.start()
                    #call thread for input
                frames = 0
            else:
                frames +=1



            # Visualize the results on the frame

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

play_video("./videos/cropped_vids/canelo2.mp4", 33)