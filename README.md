# eye_tracking_demo


### demo_v0.3.py
기준이 되는 지표 설정 후 시선에 따른 영역 분류
### demo_v0.5.py
고정된 값인 디바이스와 사용자의 거리, 디바이스 모니터 가로세로 길이 활용 - 탄젠트 값의 변화 이용
분할된 점들의 위치에 따라 탄젠트 변화값으로 1,2 / 3,4 / 5,6 영역 분류
### 캠부터 유저까지의거리.py
mediapipe에서 동공의 좌표를 이용하고 그 중 z값을 re-scale해서 사용
### 아이트래킹_마우스커서제어_v0.3.py 
광고 영상이 재생되는 동안 피험자가 화면에서 응시하는 영역 저장
15초 광고의 총 프레임 수는 450으로 fps는 정확히 30
모델 사용으로 인해 시간이 늘어져 프레임 수 카운트
