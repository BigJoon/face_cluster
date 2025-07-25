# -*- coding: utf-8 -*-

# --- 설정값 ---

# 입출력 폴더 경로
INPUT_VIDEO_DIR = "input_video"       # 분석할 동영상이 있는 폴더
OUTPUT_CLIPS_DIR = "output_clips"     # 결과물(클립, 대표얼굴)이 저장될 폴더

# 얼굴 클러스터링(DBSCAN) 설정
DBSCAN_EPS = 0.35                     # 얼굴 인코딩 간의 최대 거리. 작을수록 동일 인물에 대한 판단이 엄격해집니다.
DBSCAN_MIN_SAMPLES = 7                # 하나의 인물 클러스터를 형성하기 위해 필요한 최소 얼굴 샘플 수. 노이즈성 단일 검출을 거르는데 도움이 됩니다.

# 동영상 처리 설정
FRAME_SAMPLING_RATE = 2               # 초당 분석할 프레임 수. 높을수록 정확하지만 분석 시간이 오래 걸립니다.
CLIP_MERGE_TOLERANCE_SEC = 2.5        # 같은 인물의 등장 구간을 하나로 합칠 때 허용할 최대 시간 간격(초)입니다.