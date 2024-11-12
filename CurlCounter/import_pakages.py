import cv2
import mediapipe as mp
import numpy as np

# drawing과 관련된 유틸들
mp_drawing = mp.solutions.drawing_utils   

# pose를 감지하는 mediapipe model
# https://ai.google.dev/edge/mediapipe/solutions/guide?hl=ko
mp_pose = mp.solutions.pose  
