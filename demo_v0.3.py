## 디바이스 높이 : 113
## 디바이스와 사용자의 거리 : 약 100cm
## 사용자 키 : 167cm

import cv2
import mediapipe as mp
import math
import numpy as np
import csv
from collections import defaultdict
import time
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(1)
video_file_path = 'C:/Users/sblim/Desktop/06_여기어때 (1).mp4' # 1920x1080 -> 400x948
capture = cv2.VideoCapture(video_file_path)

user_camera_distance = 80.0

answer_counter = defaultdict(int)
frame_counter = 0
current_second = 0
fps = 30 # ?
csv_data_list = []
csv_file_path = "C:/Users/sblim/Desktop/answer_results.csv"

start_time = time.time()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    video_ret, video_frame = capture.read()

    width, height = frame.shape[1], frame.shape[0]
    cv2.line(frame, (width // 2, 0), (frame.shape[1] // 2, height), (0, 255, 0), 1)
    cv2.line(frame, (width * 3 // 4, 0), (width * 3 // 4, height), (0, 255, 0), 1)

    if not ret:
        break

    if not video_ret:
        break

    results = face_mesh.process(frame)
    
    h, w, c = video_frame.shape
    rows, cols = 3, 2
    col_width = w // cols
    for i in range(1, cols):
        video_frame[:, i * col_width] = [255, 255, 255]  # 흰색 선

    row_height = h // rows
    for i in range(1, rows):
        video_frame[i * row_height, :] = [255, 255, 255]  # 흰색 선

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        center_eye_x, center_eye_y, center_eye_z = (landmarks[159].x + landmarks[386].x) / 2, (landmarks[159].y + landmarks[386].y) / 2, (landmarks[159].z + landmarks[386].z) / 2
        nose_x, nose_y, nose_z = landmarks[4].x, landmarks[4].y, landmarks[4].z

        center_eye_z -= user_camera_distance
        nose_z -= user_camera_distance 

        gaze_vector = (center_eye_x - nose_x, center_eye_y - nose_y, center_eye_z - nose_z)

        yaw = math.atan2(gaze_vector[0], gaze_vector[2]) * 180 / math.pi
        pitch = math.atan2(-gaze_vector[1], math.sqrt(gaze_vector[0]**2 + gaze_vector[2]**2)) * 180 / math.pi

        # yaw, pitch 양수 값이면 오른쪽이나 위쪽 / 음수 값이면 왼쪽이나 아래쪽

        cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 시작 정면 기준값 : pitch 65, yaw -10
        answer = ''
        # 1,2
        if pitch < 62.1:
            if yaw > -5:
                answer = '1'
                answer_counter['1'] += 1
            else:
                answer = '2'
                answer_counter['2'] += 1
        # 3,4
        elif pitch < 66:
            if yaw > -4:         
                answer ='3'
                answer_counter['3'] += 1
            else:
                answer ='4'
                answer_counter['4'] += 1
        # 5, 6
        else: 
            if yaw > -2.5:
                answer ='5'
                answer_counter['5'] += 1
            else:
                answer ='6'
                answer_counter['6'] += 1

        video_frame = video_frame.astype('int')
        if answer == '1':
            video_frame[:row_height, :col_width, :] -= 100
        if answer == '2':
            video_frame[:row_height, col_width:, :] -= 100
        if answer == '3':
            video_frame[row_height:row_height*2, :col_width, :] -= 100
        if answer == '4':
            video_frame[row_height:row_height*2, col_width:, :] -= 100
        if answer == '5':
            video_frame[row_height*2:, :col_width, :] -= 100
        if answer == '6':
            video_frame[row_height*2:, col_width:, :] -= 100

        frame_counter += 1
        
        if frame_counter == int(fps):
            max_answer = max(answer_counter, key=answer_counter.get)
            csv_data_list.append([current_second + 1, max_answer])
            answer_counter = defaultdict(int)
            frame_counter = 0
            current_second += 1

    cv2.imshow('Eye Gaze Estimation', frame)

    video_frame = np.clip(video_frame, 0, 255).astype('uint8')
    cv2.imshow("frame", video_frame)
        
    key = cv2.waitKey(33) & 0xFF
    if key == 27:
        break

# 영상의 1초당 프레임 수 가져오기
fps = capture.get(cv2.CAP_PROP_FPS)
print("실제 1초당 프레임 수:", fps)


# 총 프레임 개수 확인
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames in the video:", total_frames)

df = pd.DataFrame(csv_data_list, columns=["Time (s)", "Most Frequent Answer"])
df.to_csv(csv_file_path, index=False)

cap.release()
cv2.destroyAllWindows()