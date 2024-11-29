import cv2
import mediapipe as mpp  # 약칭 변경
import numpy as np
from tensorflow.keras.models import load_model
import multiprocessing as mp
import time
import google.generativeai as genai


genai.configure(api_key='')

# MediaPipe 초기화
mpp_holistic = mpp.solutions.holistic
mpp_drawing = mpp.solutions.drawing_utils

# 모델 로드
model = load_model('../bestGRU_model.keras')
label_names = ['가능', '괜찮다', '기다리다', '끝', '도착', '돈', '되다', '맞다',
               '불편하다', '수고', '실종', '심하다', '영수증', '원하다', '유턴',
               '잃어버리다', '접근', '차밀리다', '카드', '필요', '화나다']

# 키포인트 추출 함수
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, left_hand, right_hand], axis=0)

# 프레임 캡처 프로세스
def capture_frames(frame_queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    holistic = mpp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠 읽기 실패.")
            break

        frame = cv2.resize(frame, (320, 240))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        # 키포인트 추출 및 변환
        keypoints = extract_keypoints(results)

        # 프레임과 키포인트를 큐에 추가
        if frame_queue.full():
            frame_queue.get()  # 큐가 꽉 차면 가장 오래된 데이터를 제거
        frame_queue.put((frame, keypoints))

    cap.release()
    holistic.close()

# 예측된 단어를 저장할 리스트와 관련 변수들
def predict_frames(frame_queue, result_queue, word_queue, fixed_length=60):
    sequence = []
    predicted_words = []
    last_predicted_label = None
    last_word_time = time.time()
    sentence_threshold = 1.5  # 초 단위, 문장 구분을 위한 시간 임계값

    while True:
        if not frame_queue.empty():
            _, keypoints = frame_queue.get()
            sequence.append(keypoints)

            if len(sequence) > fixed_length:
                sequence.pop(0)

            if len(sequence) == fixed_length:
                input_data = np.expand_dims(sequence, axis=0).reshape(1, fixed_length, -1)
                prediction = model.predict(input_data, verbose=0)
                predicted_label_idx = np.argmax(prediction)
                predicted_label = label_names[predicted_label_idx]
                confidence = np.max(prediction) * 100

                # 신뢰도 임계값 설정
                if confidence > 70:
                    # 연속적인 중복 단어 제거
                    if predicted_label != last_predicted_label:
                        predicted_words.append(predicted_label)
                        last_predicted_label = predicted_label
                        last_word_time = time.time()  # 마지막 단어 입력 시간 갱신
                        print(f"예측된 단어들: {predicted_words}")
                else:
                    last_predicted_label = None  # 신뢰도가 낮으면 이전 단어 초기화

                # 결과 큐에 추가
                if result_queue.full():
                    result_queue.get()
                result_queue.put((predicted_label, confidence))

        # 문장 구분 로직: 일정 시간 동안 새로운 단어가 없으면 문장 생성
        current_time = time.time()
        if predicted_words and (current_time - last_word_time) > sentence_threshold:
            # 문장 생성 함수 호출
            sentence = generate_sentence(predicted_words)
            print(f"생성된 문장: {sentence}")
            # 단어 리스트 초기화
            predicted_words.clear()
            last_predicted_label = None
            last_word_time = current_time
            # 생성된 문장을 다른 프로세스로 전달하고 싶다면 word_queue를 사용
            if word_queue.full():
                word_queue.get()
            word_queue.put(sentence)


# Gemini를 활용한 문장 생성 함수
def generate_sentence(predicted_words):
    try:
        # 단어들을 공백으로 연결
        joined_words = ' '.join(predicted_words)
        
        # 프롬프트 설정
        prompt = (
            f"다음 단어들은 수어로 인식된 결과입니다: {joined_words}. "
            "이 단어들을 사용해 문맥에 맞고 자연스러운 한국어 문장을 만들어주세요. "
            "결과는 단 하나의 문장으로만 작성해주세요."
        )
    
        # 생성 모델 설정
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        # 텍스트 생성
        response = model.generate_content(prompt)
        # 응답 결과 반환
        sentence = response.text.strip()
        return sentence
    except Exception as e:
        print(f"Gemini API 요청 중 오류 발생: {e}")
        return f"Gemini API 요청 중 오류 발생: {e}"


# 시각화 및 출력 프로세스 수정
def display_frames(frame_queue, result_queue, word_queue):
    prev_time = time.time()  # FPS 측정을 위한 초기값 설정

    while True:
        if not frame_queue.empty():
            frame, _ = frame_queue.get()

            # 예측 결과 표시
            if not result_queue.empty():
                predicted_label, confidence = result_queue.get()
                if confidence > 50:  # 신뢰도 임계값
                    # 프레임에 결과 출력
                    cv2.putText(frame, f"{predicted_label} ({confidence:.2f}%)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 생성된 문장 표시
            if not word_queue.empty():
                sentence = word_queue.get()
                # 프레임에 문장 출력
                cv2.putText(frame, f"생성된 문장: {sentence}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # FPS 계산 및 표시
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # 프레임 출력
            cv2.imshow('Real-time Detection', frame)

            # 'q' 키로 종료
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# 멀티프로세싱 실행
if __name__ == "__main__":
    # 큐 초기화
    frame_queue = mp.Queue(maxsize=5)
    result_queue = mp.Queue(maxsize=5)
    word_queue = mp.Queue(maxsize=5)  # 생성된 문장을 전달하기 위한 큐 추가

    # 각 프로세스 정의
    capture_process = mp.Process(target=capture_frames, args=(frame_queue,))
    predict_process = mp.Process(target=predict_frames, args=(frame_queue, result_queue, word_queue))
    display_process = mp.Process(target=display_frames, args=(frame_queue, result_queue, word_queue))

    # 프로세스 시작
    capture_process.start()
    predict_process.start()
    display_process.start()

    # 프로세스 종료 대기
    capture_process.join()
    predict_process.join()
    display_process.join()
