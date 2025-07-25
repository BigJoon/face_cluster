import face_recognition
from sklearn.cluster import DBSCAN
import numpy as np
import os
import shutil
from collections import Counter, defaultdict
import cv2
import subprocess
import sys
import config as cfg

def extract_face_data_from_video(video_path):
    """동영상에서 프레임을 샘플링하여 얼굴 인코딩, 타임스탬프, 위치를 추출합니다."""
    print(f"📹 동영상 파일 '{video_path}' 분석을 시작합니다...")
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"오류: 동영상 파일을 열 수 없습니다: {video_path}")
        return None

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / cfg.FRAME_SAMPLING_RATE)
    
    face_data = []
    frame_count = 0
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # 지정된 간격의 프레임만 처리
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            
            # BGR (OpenCV) -> RGB (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 얼굴 위치 탐지 및 인코딩
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for i, encoding in enumerate(face_encodings):
                    face_data.append({
                        "encoding": encoding,
                        "timestamp": timestamp,
                        "location": face_locations[i] # 얼굴 위치 저장
                    })
            
            # 진행 상황 표시
            sys.stdout.write(f"    - 처리 중: {int(timestamp)}초 지점, 찾은 얼굴 수: {len(face_data)}개\r")
            sys.stdout.flush()

        frame_count += 1
        
    video.release()
    print(f"\n✨ 총 {len(face_data)}개의 얼굴 데이터를 동영상에서 추출했습니다.")
    return face_data


def save_representative_faces(video_path, face_data, labels):
    """각 클러스터의 대표 얼굴 이미지를 저장합니다."""
    print("🖼️  각 그룹의 대표 얼굴을 저장합니다...")
    
    # 1. 각 클러스터(label)의 첫 번째 얼굴 데이터 찾기
    first_faces = {}
    for i, label in enumerate(labels):
        if label != -1 and label not in first_faces:
            first_faces[label] = face_data[i]

    if not first_faces:
        print("    - 대표 얼굴로 지정할 클러스터가 없습니다.")
        return

    # 2. 타임스탬프별로 정리하여 비디오를 한 번만 탐색하도록 최적화
    faces_by_timestamp = defaultdict(list)
    for label, data in first_faces.items():
        faces_by_timestamp[data['timestamp']].append({'label': label, 'location': data['location']})

    # 3. 비디오를 열고 프레임을 탐색하며 얼굴 이미지 저장
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    for timestamp in sorted(faces_by_timestamp.keys()):
        frame_num = int(timestamp * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video.read()

        if ret:
            for face_info in faces_by_timestamp[timestamp]:
                label = face_info['label']
                top, right, bottom, left = face_info['location']
                
                face_image = frame[top:bottom, left:right]
                
                person_dir = os.path.join(cfg.OUTPUT_CLIPS_DIR, f"person_{label}")
                os.makedirs(person_dir, exist_ok=True)
                
                output_path = os.path.join(person_dir, "representative_face.jpg")
                cv2.imwrite(output_path, face_image)
                print(f"    - 'person_{label}' 그룹의 대표 얼굴 저장 완료.")

    video.release()


def generate_timelines(face_data, labels):
    """클러스터링 결과를 바탕으로 인물별 등장 타임라인을 생성하고 병합합니다."""
    print("⏳ 인물별 등장 타임라인을 생성합니다...")
    
    timelines = defaultdict(list)
    for i, data in enumerate(face_data):
        label = labels[i]
        if label != -1:
            timelines[label].append(data['timestamp'])

    merged_timelines = defaultdict(list)
    for label, timestamps in timelines.items():
        timestamps.sort()
        
        if not timestamps:
            continue
            
        current_start = timestamps[0]
        current_end = timestamps[0]
        
        for i in range(1, len(timestamps)):
            if timestamps[i] - current_end <= cfg.CLIP_MERGE_TOLERANCE_SEC:
                current_end = timestamps[i]
            else:
                if current_end - current_start > 1.0:
                    merged_timelines[label].append((current_start, current_end))
                current_start = timestamps[i]
                current_end = timestamps[i]
        
        if current_end - current_start > 1.0:
             merged_timelines[label].append((current_start, current_end))

    print("✅ 타임라인 생성 완료!")
    return merged_timelines


def create_clips_from_timelines(video_path, timelines):
    """생성된 타임라인에 따라 FFmpeg를 사용하여 동영상 클립을 생성합니다."""
    print("🎬 동영상 클립 생성을 시작합니다...")
    
    for label, intervals in timelines.items():
        person_dir = os.path.join(cfg.OUTPUT_CLIPS_DIR, f"person_{label}")
        os.makedirs(person_dir, exist_ok=True) # 디렉토리는 이미 대표 얼굴 저장 시 생성되었을 수 있음
        print(f"  - 'person_{label}' 그룹의 클립 생성 중...")
        
        for i, (start, end) in enumerate(intervals):
            output_filename = os.path.join(person_dir, f"clip_{i+1}.mp4")
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start),
                '-to', str(end),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-y',
                '-loglevel', 'error',
                output_filename
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except FileNotFoundError:
                print("\n❌치명적 오류: 'ffmpeg'을 찾을 수 없습니다. 시스템에 FFmpeg이 설치되어 있고, PATH에 등록되어 있는지 확인해주세요.")
                return
            except subprocess.CalledProcessError as e:
                print(f"\n⚠️ FFmpeg 실행 중 오류 발생: {e}")

    print("\n🎉 모든 작업이 완료되었습니다! 'output_clips' 폴더를 확인해보세요.")


def process_video(video_path):
    """메인 파이프라인: 동영상 처리, 클러스터링, 클립 생성을 총괄합니다."""
    
    face_data = extract_face_data_from_video(video_path)
    if not face_data or len(face_data) < cfg.DBSCAN_MIN_SAMPLES:
        print("🤷‍♀️ 얼굴을 충분히 찾지 못해 클러스터링을 진행할 수 없습니다.")
        return

    encodings = [d['encoding'] for d in face_data]
    print(f"\n🤖 총 {len(encodings)}개의 얼굴을 찾았습니다. 클러스터링을 시작합니다...")
    
    clt = DBSCAN(metric="euclidean", eps=cfg.DBSCAN_EPS, min_samples=cfg.DBSCAN_MIN_SAMPLES)
    clt.fit(encodings)
    
    labels = clt.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = np.sum(np.array(labels) == -1)
    print(f"✅ 클러스터링 완료! 총 {num_clusters}개의 인물 그룹과 {noise_count}개의 분류되지 않은 얼굴을 찾았습니다.")

    # 출력 폴더 초기화
    if os.path.exists(cfg.OUTPUT_CLIPS_DIR):
        shutil.rmtree(cfg.OUTPUT_CLIPS_DIR)
    os.makedirs(cfg.OUTPUT_CLIPS_DIR)

    # 대표 얼굴 저장
    save_representative_faces(video_path, face_data, labels)

    # 타임라인 생성
    timelines = generate_timelines(face_data, labels)
    if not timelines:
        print("🤷‍♂️ 클립을 생성할 유의미한 등장 구간을 찾지 못했습니다.")
        return

    # 클립 생성
    create_clips_from_timelines(video_path, timelines)


if __name__ == "__main__":
    if not os.path.exists(cfg.INPUT_VIDEO_DIR):
        os.makedirs(cfg.INPUT_VIDEO_DIR)
        print(f"폴더 생성: '{cfg.INPUT_VIDEO_DIR}'. 이 폴더에 분석할 동영상을 넣어주세요.")
        sys.exit()

    video_files = [f for f in os.listdir(cfg.INPUT_VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    
    if not video_files:
        print(f"분석할 동영상이 없습니다. '{cfg.INPUT_VIDEO_DIR}' 폴더에 동영상 파일을 넣어주세요.")
        sys.exit()

    video_to_process = os.path.join(cfg.INPUT_VIDEO_DIR, video_files[0])
    
    process_video(video_to_process)