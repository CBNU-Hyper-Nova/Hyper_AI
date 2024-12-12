from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import numpy as np
import google.generativeai as genai
import time

app = Flask(__name__)
CORS(app)  # 프론트엔드에서의 요청을 허용

# Gemini API 설정
genai.configure(api_key='')  # 실제 API 키로 대체

# 커스텀 레이어 정의
class GetItem(Layer):
    def __init__(self, index, **kwargs):
        super(GetItem, self).__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index]

    def get_config(self):
        config = super(GetItem, self).get_config()
        config.update({"index": self.index})
        return config

# 모델 로드
model_path = '../model/transformer_sign_language_model.h5'
model = load_model(model_path, custom_objects={'GetItem': GetItem})

label_names = [
    "10분",
    "119",
    "1호",
    "1회",
    "2호",
    "3시",
    "40분",
    "4사람",
    "5분",
    "8호",
    "9호",
    "가능",
    "가다",
    "갈아타다",
    "감사합니다",
    "건너다",
    "경찰",
    "고속터미널",
    "고장",
    "곳",
    "곳곳",
    "공기청정기",
    "괜찮다",
    "교통카드",
    "교환하다",
    "국립박물관",
    "그만",
    "급하다",
    "기다리다",
    "길",
    "까먹다",
    "끄다",
    "끝",
    "나",
    "나르다",
    "나사렛",
    "난방",
    "내리다",
    "냄새",
    "늦다",
    "다시",
    "다음",
    "단말기터치",
    "당신",
    "대로",
    "도와주다",
    "도움받다",
    "도착",
    "돈",
    "돈얼마",
    "돈주다",
    "되다",
    "들어올리다",
    "딱",
    "떨어지다",
    "마포대교",
    "막차",
    "만원",
    "맞다",
    "명동",
    "몇분",
    "몇사람",
    "몇호",
    "모르다",
    "목적",
    "무엇",
    "문자받다",
    "물품보관",
    "미안합니다",
    "반갑다",
    "받다",
    "발생하다",
    "방법",
    "방황",
    "백화점",
    "버스",
    "번호",
    "보건소",
    "보다",
    "부르다",
    "불가능",
    "불량",
    "불편하다",
    "빨리",
    "뼈곳",
    "사거리",
    "사다",
    "샛길",
    "서대문농아인복지관",
    "서울농아인협회",
    "서울역",
    "수고",
    "시간",
    "시청",
    "신분당",
    "신분증",
    "신호등",
    "실종",
    "심하다",
    "쓰러지다",
    "아니다",
    "아직",
    "아프다",
    "안내소",
    "안내하다",
    "안녕하세요",
    "안되다",
    "안전벨트",
    "알다",
    "알려받다",
    "알려주다",
    "어떻게",
    "어렵다",
    "어린이교통카드",
    "어린이집",
    "언덕",
    "얼마",
    "없다",
    "에어컨",
    "엘리베이터",
    "여기",
    "역무원",
    "연착",
    "영수증",
    "오다오다",
    "오른쪽",
    "오천원",
    "오케이",
    "올리다",
    "옷가게",
    "왜",
    "왼쪽",
    "용산역",
    "우회전",
    "원래",
    "원하다",
    "위아래",
    "위험",
    "유턴",
    "육교",
    "응급실",
    "일정하다",
    "잃어버리다",
    "있다",
    "자판기",
    "잘못",
    "잘못하다",
    "잠실대교",
    "장애인복지카드",
    "저기",
    "전",
    "전화걸다",
    "접근",
    "정기권",
    "조심",
    "좌회전",
    "주세요",
    "주차",
    "죽다",
    "중",
    "지름길",
    "지연되다",
    "지하철",
    "짐",
    "차내리다",
    "차두다",
    "차따라가다",
    "차밀리다",
    "찾다",
    "천안아산역",
    "천천히",
    "철로",
    "첫차",
    "청음회관",
    "충분",
    "충분하다",
    "충전",
    "카드",
    "카톡보내다",
    "켜다",
    "타다",
    "트렁크닫다",
    "트렁크열다",
    "틀리다",
    "파리바게트",
    "편의점",
    "편지",
    "표",
    "표지판",
    "필요",
    "필요없다",
    "한국농아인협회",
    "항상",
    "해보다",
    "화나다",
    "확인",
    "확인증",
    "회의실",
    "횡단보도",
    "힘들다"
]

# 버퍼 초기화
buffer = {
    'keypoints': [],
    'timestamps': [],
    'detected_words': [],  # 단어 버퍼
    'last_processed_time': 0  # 마지막 문장 생성 시각
}

FRAME_COUNT = 30  # 모델이 요구하는 프레임 수
CONFIDENCE_THRESHOLD = 0.6  # 신뢰도 기준값
TIMEOUT_SECONDS = 5  # 타임아웃 (2초 동안 추가 단어가 없으면 문장 생성)

@app.route('/process-keypoints', methods=['POST'])
def process_keypoints():
    global buffer
    data = request.get_json()
    frame_num = data.get('frame_num', None)
    keypoints = data.get('keypoints', [])
    timestamp = data.get('timestamp', None)

    # 데이터 유효성 검증
    if frame_num is None or not keypoints or timestamp is None:
        return jsonify({'error': 'Invalid data provided'}), 400

    if len(keypoints) != 225:
        return jsonify({'error': f'Invalid number of keypoints. Expected 225, got {len(keypoints)}'}), 400

    try:
        # 키포인트 버퍼에 추가
        buffer['keypoints'].append(keypoints)
        buffer['timestamps'].append(timestamp)

        # 30 프레임이 쌓였을 때 예측 수행
        if len(buffer['keypoints']) >= FRAME_COUNT:
            batch_keypoints = buffer['keypoints'][:FRAME_COUNT]
            batch_timestamps = buffer['timestamps'][:FRAME_COUNT]

            # 버퍼에서 처리된 데이터 제거
            buffer['keypoints'] = buffer['keypoints'][FRAME_COUNT:]
            buffer['timestamps'] = buffer['timestamps'][FRAME_COUNT:]

            # 전처리: (1, FRAME_COUNT, 225)
            batch_keypoints_np = np.array(batch_keypoints, dtype=np.float32)
            batch_keypoints_np = batch_keypoints_np.reshape(1, FRAME_COUNT, 225)

            # 모델 예측
            predictions = model.predict(batch_keypoints_np)
            for pred in predictions:
                if max(pred) > CONFIDENCE_THRESHOLD:
                    predicted_word = label_names[np.argmax(pred)]
                    buffer['detected_words'].append(predicted_word)

        # 2초 동안 새로운 단어가 없으면 문장 생성
        current_time = time.time()
        if current_time - buffer['last_processed_time'] > TIMEOUT_SECONDS and buffer['detected_words']:
            # 중복 제거
            unique_labels = remove_consecutive_duplicates(buffer['detected_words'])

            # 문장 생성
            final_sentence = generate_sentence(unique_labels)

            # 문장 생성 후 버퍼 초기화
            buffer['detected_words'] = []
            buffer['last_processed_time'] = current_time

            response = {
                'timestamp': timestamp,
                'sentence': final_sentence
            }
            return jsonify(response), 200

        # 단순 대기 상태 반환
        return jsonify({'status': 'waiting', 'buffer_length': len(buffer['keypoints'])}), 200

    except Exception as e:
        print(f"Error processing keypoints: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def remove_consecutive_duplicates(labels):
    """
    연속적으로 중복된 단어를 제거합니다.
    """
    unique_labels = []
    prev_label = None
    for label in labels:
        if label != prev_label:
            unique_labels.append(label)
            prev_label = label
    return unique_labels

def generate_sentence(predicted_labels):
    try:
        # 예측된 라벨들을 공백으로 연결
        joined_words = ' '.join(predicted_labels)

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
        return "수어를 인식 중 입니다."

if __name__ == '__main__':
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5001, debug=True)
