## 디바이스 높이 : 132
## 디바이스와 사용자의 거리 : 약 80cm
## 사용자 키 : 167cm

# 탄젠트 벡터를 이용해서 해보기
import cv2
import mediapipe as mp
import math
import numpy as np
import csv
from collections import defaultdict
import time
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 카메라와 디바이스 실제 거리
user_camera_distance = 80.0
# 실제 디바이스에서 분할 유닛 한개 실제 사이즈 
device_sec_height = 19.0
device_sec_width = 12.2

tan_theta_1 = device_sec_height / user_camera_distance
theta_radians_1 = math.atan(tan_theta_1)
theta_angle_1 = math.degrees(theta_radians_1)

tan_theta_2 = device_sec_height * 2 / user_camera_distance
theta_radians_2 = math.atan(tan_theta_2)
theta_angle_2 = math.degrees(theta_radians_2)

cap = cv2.VideoCapture(1)  

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    width, height = frame.shape[1], frame.shape[0]
    
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(frame_rgb)

    if results_mesh.multi_face_landmarks:
        landmarks = results_mesh.multi_face_landmarks[0].landmark

        # 코 좌표(각 구간마다 기준이 될 랜드마크 좌표)
        # 코에 인덱스 4
        mesh_coords = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in results_mesh.multi_face_landmarks[0].landmark]
        mesh_points = [mesh_coords[p] for p in [4]]

        p = mesh_points[0]
        cv2.circle(frame, p, 4, (0, 255, 0), -1)

        # 기준점
        # 캠에서 잡힌 얼굴의 코 좌표
        center = (393, 253) # 253~255
        cv2.circle(frame, center, 4, (255, 0, 0), -1)

        radius = 3 
        if math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2) < radius :
            print('원 안에 있습니다. 다음 단계로 진행하세요')
            break
            # 이 때가 이제 세타 값이 0
    cv2.imshow("frame", frame)

    # waitKey와 destroyAllWindows 추가


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    width, height = frame.shape[1], frame.shape[0]
    
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(frame_rgb)

    if results_mesh.multi_face_landmarks:
        landmarks = results_mesh.multi_face_landmarks[0].landmark

        # 코 좌표(각 구간마다 기준이 될 랜드마크 좌표)
        mesh_coords = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in results_mesh.multi_face_landmarks[0].landmark]
        mesh_points = [mesh_coords[p] for p in [4]]

        p = mesh_points[0]
        cv2.circle(frame, p, 4, (0, 255, 0), -1)
        # print(p)

        # 기준점
        center = (393, 253) # 253~255
        cv2.circle(frame, center, 4, (255, 0, 0), -1)
        theta = 0

        p1 = (393, 243)
        p2 = (393, 247)
        p3 = (394, 256)
        cv2.circle(frame, p1, 4, (0, 0, 255), -1)
        cv2.circle(frame, p2, 4, (0, 0, 255), -1)
        cv2.circle(frame, p3, 4, (0, 0, 255), -1)

        # y 좌표 1 증가당 세타값 증가량
        up_ratio = theta_angle_1 / (center[1] - p2[1])
        user_theta = (center[1] - p[1]) * up_ratio

        cv2.putText(frame, f'theta: {user_theta:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if user_theta >= theta_angle_1:
            answer = '1, 2'
        else:
            if user_theta >= 0:
                answer = '3, 4'
            else:
                answer = '5, 6'

        print(answer)
        cv2.putText(frame, f'answer: {answer}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # 'ESC' 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()