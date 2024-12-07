# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
import time
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Gemini API 키 설정
genai.configure(api_key='')

# 모델 로드 (모델 경로 수정 필요)
model = load_model('../model/bestGRU_model.keras')
label_names = ['가능', '괜찮다', '기다리다', '끝', '도착', '돈', '되다', '맞다',
               '불편하다', '수고', '실종', '심하다', '영수증', '원하다', '유턴',
               '잃어버리다', '접근', '차밀리다', '카드', '필요', '화나다']

@app.route('/process-keypoints', methods=['POST'])
def process_keypoints():
    data = request.get_json()
    timestamp = data.get('timestamp', None)
    coords = data.get('data', [])

    print("------ 요청 수신 ------")
    print(f"타임스탬프: {timestamp}")
    print(f"좌표 데이터 길이: {len(coords)}")

    if not coords:
        print("좌표 데이터 없음")
        return jsonify({'error': 'No coordinates provided'}), 400

    # 여기서는 pose+left_hand+right_hand 총 75개 랜드마크 * 60프레임 = 4500개 포인트 가정
    # coords는 frame_num에 따라 정렬되어 있다고 가정
    # 실제 구현 시 frame_num과 landmark_type, index를 기준으로 정렬 필요
    # 여기서는 단순히 coords가 이미 올바른 순서로 들어온다고 가정

    # x,y,z만 추출
    xyz_list = [(c['x'], c['y'], c['z']) for c in coords]
    xyz_array = np.array(xyz_list)
    # (N,3) 형태. N이 60프레임*75랜드마크=4500이라고 가정
    frames = 60
    landmarks_per_frame = 75

    if xyz_array.shape[0] != frames * landmarks_per_frame:
        print(f"입력 데이터 크기 불일치: 예상 {frames*landmarks_per_frame}, 실제 {xyz_array.shape[0]}")
        return jsonify({'error': 'Input data size does not match expected frames*landmarks'}), 400

    xyz_array = xyz_array.reshape(frames, landmarks_per_frame*3)  # (60,225)
    xyz_array = xyz_array[np.newaxis, ...]  # (1,60,225)

    print("모델 예측 시작...")
    predictions = model.predict(xyz_array)
    predicted_label_idx = np.argmax(predictions)
    predicted_label = label_names[predicted_label_idx]
    confidence = float(np.max(predictions) * 100)
    print(f"예측 완료: {predicted_label}, 신뢰도: {confidence:.2f}%")

    predicted_words = [predicted_label]
    print("문장 생성 중...")
    sentence = generate_sentence(predicted_words)
    print(f"생성된 문장: {sentence}")

    response = {
        'timestamp': timestamp,
        'sentence': sentence,
        'confidence': confidence
    }

    print("응답 전송:", response)
    return jsonify(response)

def generate_sentence(predicted_words):
    try:
        joined_words = ' '.join(predicted_words)
        prompt = (
            f"다음 단어들은 수어로 인식된 결과입니다: {joined_words}. "
            "이 단어들을 사용해 문맥에 맞고 자연스러운 한국어 문장을 만들어주세요. "
            "결과는 단 하나의 문장으로만 작성해주세요."
        )

        # Gemini 모델 호출 (API 키 설정 필요)
        # genai.configure(api_key='YOUR_API_KEY_HERE')
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        sentence = response.text.strip()
        return sentence
    except Exception as e:
        print(f"Gemini API 요청 중 오류 발생: {e}")
        return f"Gemini API 요청 중 오류 발생: {e}"

if __name__ == '__main__':
    print("서버 시작 중...")
    app.run(host='0.0.0.0', port=5001, debug=True)
