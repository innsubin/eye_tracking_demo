import time

import cv2
import pandas as pd
import mediapipe as mp
import pyautogui


# 캠 불러오기
capture_webcam = cv2.VideoCapture(0)
capture_video = cv2.VideoCapture('06_여기어때.mp4')

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# 조정된 w, h
adj_screen_w, adj_screen_h = 4000, 6400

# 원본 w, h
screen_w, screen_h = pyautogui.size()
screen_w, screen_h = int(str(screen_w)), int(str(screen_h))

# 창 생성
cv2.namedWindow('ad', cv2.WINDOW_NORMAL)

# 창 위치 지정
cv2.moveWindow('ad', 0, 0)

# 결과 저장 리스트
# 1초동안 피험자가 가장 많이 본 영역 저장
focus = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0}
result_focus = []
current_status = '-1'
frame_count = 0
while True:
    ret_webcam, frame_webcam = capture_webcam.read()
    ret_video, frame_video = capture_video.read()
    
    if not ret_webcam or not ret_video:
        break

    frame_webcam = cv2.flip(frame_webcam, 1)
    
    # 비디오 높이를 목표 높이에 맞추고, 비율에 따라 너비 조정
    scale_ratio = screen_h / frame_video.shape[0]
    new_width = int(frame_video.shape[1] * scale_ratio)

    # 비디오 리사이즈
    resized_frame = cv2.resize(frame_video, (new_width, screen_h))
    
    # 좌우 패딩 계산
    padding_left = int((screen_w - new_width) / 2)
    padding_right = screen_w - new_width - padding_left

    # 검은색 패딩 추가
    padded_frame = cv2.copyMakeBorder(resized_frame, 0, 0, padding_left, padding_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    rgb_frame = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    
    frame_h, frame_w, _ = frame_webcam.shape # frame : 480x640x3
    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        # 많은 랜드마크 중 4개의 오른쪽 눈 랜드마크만을 표시
        x = landmarks[474].x
        y = landmarks[474].y
                
        screen_x = int(adj_screen_w * x - 1200) 
        screen_y = int(adj_screen_h * y - 2500)

        # 1번 컬럼
        if screen_x < padding_left + new_width // 2:
            if screen_y < screen_h // 3:
                current_status = '1'
                focus['1'] += 1
            elif screen_y < (screen_h // 3) * 2:
                current_status = '3'
                focus['3'] += 1
            else:
                current_status = '5'
                focus['5'] += 1
        # 2번 컬럼
        else:
            if screen_y < screen_h // 3:
                current_status = '2'
                focus['2'] += 1
            elif screen_y < (screen_h // 3) * 2:
                current_status = '4'
                focus['4'] += 1
            else:
                current_status = '6'
                focus['6'] += 1
        cv2.putText(padded_frame, current_status, (30, 100), 1, 5, (0, 255, 0), 3)
        cv2.circle(padded_frame, (screen_x, screen_y), 3, (0, 255, 0), -1)

    cv2.imshow('ad', padded_frame)
    frame_count += 1

    # 30프레임마다 가장 많이 본 영역 저장
    # 딕셔너리 초기화
    if frame_count == 30:
        max_key = max(focus, key=focus.get)
        result_focus.append(max_key)
        focus = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0}
        frame_count = 0

    if cv2.waitKey(1) & 0xFF == 27:  # 'ESC' 키를 누르면 종료
        break

capture_webcam.release()
cv2.destroyAllWindows()

df = pd.DataFrame({'index':list(range(1, 16)), 'area':result_focus})
df.to_csv('result.csv', index=False)