# Define the path to the video file
video_path = "dancingman.mp4"

def detect_shoplifting(video_path):
    # Load YOLOv8 model (replace with the actual path to your YOLOv8 model)
    model_yolo = YOLO('yolov8s-pose.pt')

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print(f"Total Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_tot = 0
    count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Warning: Frame could not be read. Skipping.")
            break  # Stop the loop if no frame is read

        count += 1
        if count % 3 != 0:
            continue

        # Resize the frame
        frame = cv2.resize(frame, (640, 480))

        # Run YOLOv8 on the frame
        results = model_yolo(frame, verbose=False)

        # Visualize the YOLO results on the frame
        annotated_frame = results[0].plot(boxes=True)

        for r in results:
            bound_box = r.boxes.xyxy  # Bounding box coordinates
            conf = r.boxes.conf.tolist()  # Confidence levels
            keypoints = r.keypoints.xyn.tolist()  # Keypoints for human pose
            print(keypoints)

            print(f'Frame {frame_tot}: Detected {len(bound_box)} bounding boxes')

   
        # Show the annotated frame in a window
        cv2.imshow('Frame', annotated_frame)

       

        # Press 'q' to stop the video early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    

    # Close all OpenCV windows after processing is complete
    cv2.destroyAllWindows()

# Call the function with the video path
detect_shoplifting(video_path)
