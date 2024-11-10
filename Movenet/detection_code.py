# 1. Load Model
interpreter = tf.lite.Interpreter(model_path='1.tflite')
interpreter.allocate_tensors()

# 2. Make Detections
cap = cv2.VideoCapture('dancingman.mp4')  # 0: rel-time web cam 

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

window_width = 640
window_height = 480

aspect_ratio = original_width / original_height

if original_width > original_height:
    new_width = window_width
    new_height = int(window_width / aspect_ratio)
else:
    new_height = window_height
    new_width = int(window_height * aspect_ratio)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:  # 영상 끝에 도달했을 때
        break

    # Reshpae image (input: float32 256x256x3)    
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)   
    input_image = tf.cast(img, dtype=tf.uint8)


    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # get_input_details reconstruct
    is_dynamic_shape_model = input_details[0]['shape_signature'][2] == -1
    if is_dynamic_shape_model:
        input_tensor_index = input_details[0]['index']
        input_shape = input_image.shape
        interpreter.resize_tensor_input(
            input_tensor_index, input_shape, strict=True)
    interpreter.allocate_tensors()


    
    # Make predictions
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))   
    interpreter.invoke()   # prediction
    Keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])  
    print(Keypoints_with_scores)


  
    # frame resize
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # 윈도우 크기 맞추기 위해 패딩 추가
    top = (window_height - new_height) // 2
    bottom = window_height - new_height - top
    left = (window_width - new_width) // 2
    right = window_width - new_width - left

    # 패딩 추가
    frame_padded = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


    
    # Rendering
    draw_connections(frame_padded, Keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame_padded, Keypoints_with_scores, 0.4)


    cv2.imshow('MoveNet Lightning', frame_padded)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 3. Draw Keypoints function
def draw_keypoints(frame, keypoints, confidence_threshold):  # keypoints: (1x6x56)
    y, x, c = frame.shape
    shaped = []
    for i in range(0,55-5+1,3):
        shaped.append(np.array([keypoints[0][0][i]*y, keypoints[0][0][i+1]*x, keypoints[0][0][i+2]]))

    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 2, (0,255,0), -1) 

# 4. Draw Edges function
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = []
    for i in range(0,55-5+1,3):
        shaped.append(np.array([keypoints[0][0][i]*y, keypoints[0][0][i+1]*x, keypoints[0][0][i+2]]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)
