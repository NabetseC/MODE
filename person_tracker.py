import cv2
import threading

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n-pose.pt")

# Open the video file


#Helper functions:

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
                            keypoints = person.xy
                            left_hip = keypoints[0][11]
                            x, y = left_hip[0], left_hip[1]
                            xs, ys, xb, yb = boxes.xyxy[0][0], boxes.xyxy[0][1], boxes.xyxy[0][2], boxes.xyxy[0][3]
                            if x>= xs and x<=xb and y>= ys and y<= yb:
                                starting_boxes[int(boxes.id[i])].append(keypoints)
 

    """
    for frame in results_buffer:
        user_input = input("Classify \n 1:jab | 2:straight | 3:lefthook | ...")

        #save data at corresponding location
    """

    for key in starting_boxes:
        if len(starting_boxes[key]) != 40:
            continue
        else:
            print("Salutations, please classify or discard boxer ", key )
            user_input = input("[label rules]")
            print(f"You entered: {user_input}")
            """
                Here , depending on user input, save angles from the boxer to some local folder
            """



    frames = frames - 40
    cycles += 1
    start +=  40
    pending = False
    results_buffer = []

# Loop through the video frames

#set_start if you want to skip to some point the fight when starting progrm
def play_video(video_path, set_start = 0):
    global start
    global pending 
    pending = False
    start = set_start
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

play_video("./videos/cropped_vids/canelo2.mp4")