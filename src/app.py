# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # 프론트엔드에서의 요청을 허용

# Gemini API 설정
genai.configure(api_key='')  # 실제 API 키로 대체

# 모델 로드
model = load_model('../model/bestGRU_model.keras')
label_names = ['가능', '괜찮다', '기다리다', '끝', '도착', '돈', '되다', '맞다',
               '불편하다', '수고', '실종', '심하다', '영수증', '원하다', '유턴',
               '잃어버리다', '접근', '차밀리다', '카드', '필요', '화나다']

# 버퍼 초기화
buffer = {
    'keypoints': [],
    'timestamps': []
}

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

        if len(buffer['keypoints']) >= 120:
            # 버퍼에서 120 프레임 추출
            batch_keypoints = buffer['keypoints'][:120]
            batch_timestamps = buffer['timestamps'][:120]

            # 버퍼에서 추출한 데이터 제거
            buffer['keypoints'] = buffer['keypoints'][120:]
            buffer['timestamps'] = buffer['timestamps'][120:]

            # 전처리: (1, 120, 225)
            batch_keypoints_np = np.array(batch_keypoints, dtype=np.float32)
            batch_keypoints_np = batch_keypoints_np.reshape(1, 120, 225)  # (batch, time_steps, features)

            # 모델 예측
            predictions = model.predict(batch_keypoints_np)
            predicted_labels = [label_names[np.argmax(pred)] for pred in predictions]

            # Gemini를 활용한 문장 생성
            final_sentence = generate_sentence(predicted_labels)

            # 응답 데이터: 마지막 타임스탬프
            response = {
                'timestamp': batch_timestamps[-1],
                'sentence': final_sentence
            }

            return jsonify(response), 200

        else:
            # 아직 120 프레임이 모이지 않았을 때는 성공 응답만 보냄
            return jsonify({'status': 'waiting', 'buffer_length': len(buffer['keypoints'])}), 200

    except Exception as e:
        print(f"Error processing keypoints: {e}")
        return jsonify({'error': 'Internal server error'}), 500

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

    except AttributeError as e:
        print(f"Gemini API 요청 중 AttributeError 발생: {e}")
        return "문장을 생성하는 중 오류가 발생했습니다. (AttributeError)"
    except Exception as e:
        print(f"Gemini API 요청 중 오류 발생: {e}")
        return "문장을 생성하는 중 오류가 발생했습니다."

if __name__ == '__main__':
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5001, debug=True)
