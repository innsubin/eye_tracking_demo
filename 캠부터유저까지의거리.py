import time
import warnings
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

warnings.filterwarnings(action='ignore')

# mediapipe 모델 로드
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 내장 웹캠 인식
# 외장 웹캠 사용시 1로 변경
cap = cv2.VideoCapture(0)
TOT = []

while True:
    # frame 로드
    ret, frame = cap.read()
    if not ret:
        break
    
    # 좌우반전
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # 포커스미디어 디바이스 비율에 맞게 조정
    frame = frame[:, int(w*0.342):w-int(w*0.342)]
    h, w, c = frame.shape
    
    # 왼쪽 동공 x좌표의 최대값, 최소값 저장할 변수
    irisLeftMinX = -1
    irisLeftMaxX = -1

    # mediapipe 모델 사용을 위한 채널 변경
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(frame_rgb)

    # 모델 결과가 있을 때 실행
    if results_mesh.multi_face_landmarks:
        # mediapipe의 Landmark 객체를 python의 list 객체로 변환
        landmarks = results_mesh.multi_face_landmarks[0].landmark

        # 왼쪽 동공의 인덱스를 이용한 순회
        # 왼쪽 동공 x좌표의 최대값, 최소값 저장
        for i in range(468, 473):
            p = (landmarks[i].x, landmarks[i].y, landmarks[i].z)

            if irisLeftMinX == -1 or p[0] * w < irisLeftMinX:
                irisLeftMinX = p[0] * w
            
            if irisLeftMaxX == -1 or p[0] * w > irisLeftMaxX:
                irisLeftMaxX = p[0] * w

            # 왼쪽 동공 draw
            cv2.circle(frame, (int(p[0]*w),int(p[1]*h)), 4, (0, 0, 255), -1)

        # z값 re-scale 과정
        dx = irisLeftMaxX - irisLeftMinX
        dX = 11.7
        normalizedFocaleX = 1.40625
        fx = min(w, h) * normalizedFocaleX
        dZ = (fx * (dX / dx)) / 10.0
        dZ = round(dZ, 2)

        # frame에 거리 예측값 draw
        cv2.putText(frame, f'distance: {dZ}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # frame 출력
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # 'ESC' 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()