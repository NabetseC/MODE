import cv2
import threading

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file


#Helper functions:

def play_video_loop(video_path, start):
    global exit_flag
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    mf = 0

    if not cap.isOpened():
        print("Error opening video file")
        return

    while not exit_flag:
        ret, frame = cap.read()
        mf += 1
        if mf == 40:
            # Restart video
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            continue

        cv2.imshow("Video Loop", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            exit_flag = True
            break

def get_user_input():
    global start
    global pending 
    global results_buffer 
    """
    for frame in results_buffer:
        user_input = input("Classify \n 1:jab | 2:straight | 3:lefthook | ...")

        #save data at corresponding location
    """
    user_input = input("testing threading!")
    print(f"You entered: {user_input}")
    start +=  40
    pending = False

# Loop through the video frames

#set_start if you want to skip to some point the fight when starting progrm
def play_video(video_path, set_start = 0):
    global start
    global pending 
    pending = False
    start = set_start
    cap = cv2.VideoCapture(video_path)
    frames = 0

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

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)
            if not pending:
                results_buffer.append(results)

            if frames % 40  == 0 and frames > 0:
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

play_video("./videos/cropped_vids/canelo1.mp4")