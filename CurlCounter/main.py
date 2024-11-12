def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])    # 1: y, 0: x
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle

    return angle

# VIDEO FEED
cap = cv2.VideoCapture(0)  # real-time video capture (0: web cam)

# Curl Counter variables
counter = 0
stage = None

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()   # ret: 사용 안하는 값 frame: 웹캠에서 얻는 이미지

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # opencv에서는 rgb가 아닌 gbr로 처리함
        image.flags.writeable = False

        # Make detection
        # reuslts.landmarks = 각 관절 포인트의 위치 벡터 저장
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark    # 랜드마크 추출 (0~32번)

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle
            # 스크린에 rendering
            cv2.putText(image, str(angle), 
                       tuple(np.multiply(elbow, [640,480]).astype(int)),      # multiplye(elbow, [640,480]) : normalized된 elbow위치를 window크기로 맞춰줌 -> angle이 elbow위치에 나타나도록
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter += 1
                print(counter)

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        # Rep data
        cv2.putText(image, "REPS", (15,12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, "STAGE", (65,12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60,60),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


        # Render detections
        # drawing_landmarks: mp.solutions.drawing_utils.daw_landmarks를 image에 그려줌
        # POSE_CONNECTIONS: landmarks(관절 point)간의 연결 관계를 알려줌
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),  # dots drawing specification 
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)   # connections drawing specification 
                                 )
        

        # image = cv2.flip(image,1)
        cv2.imshow("Mediapipe Feed", image)  # 팝업 스크린에 웹캠을 띄워줌
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
