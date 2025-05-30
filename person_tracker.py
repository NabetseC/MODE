import cv2

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "./videos/cropped_vids/canelo1.mp4"
cap = cv2.VideoCapture(video_path)
frames = 0
first_frame = 0

new_frames = True


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        if new_frames:
            frames +=1
            if frames % 40 == 0:
                new_frames = False
        
        #UGLY if statemnts but ok for now 
        if not new_frames:
            first_frame = frames - 40

            #ask user to discard/classify punch for every person detected
            #once done, set new_frames to true and continue video from frame first_frame+40 (next vid begins at firm_frame+40)


        results = model.track(frame, persist=True, conf = .6)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()