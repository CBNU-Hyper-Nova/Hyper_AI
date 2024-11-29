import os

# TensorFlow 로그 레벨 설정 (INFO 메시지 출력)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: 모든 로그, 1: INFO, 2: WARNING, 3: ERROR만 출력

import tensorflow as tf

# 디바이스 배치 로그 활성화 (원하지 않으면 주석 처리 가능)
# tf.debugging.set_log_device_placement(True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # 가비지 컬렉션을 위해
from tqdm import tqdm  # 진행 상태 표시를 위해

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Masking, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

import concurrent.futures  # 병렬 처리를 위해
import multiprocessing  # CPU 코어 수를 얻기 위해

# -----------------------------
# 1. 데이터 로딩 및 시퀀스 생성 함수 정의
# -----------------------------

def process_csv(file_path, sequence_length=30):
    """
    단일 CSV 파일을 읽고 시퀀스를 생성합니다.

    Parameters:
    - file_path (str): CSV 파일의 경로.
    - sequence_length (int): 시퀀스당 프레임 수.

    Returns:
    - sequences (list of np.ndarray): 생성된 시퀀스 리스트.
    - labels (list): 시퀀스에 해당하는 라벨 리스트.
    """
    try:
        # 필요한 열만 선택하고 데이터 타입 최적화
        df = pd.read_csv(
            file_path,
            usecols=['frame_num', 'landmark_type', 'index', 'x', 'y', 'z', 'label'],
            dtype={
                'frame_num': 'int32',
                'landmark_type': 'category',
                'index': 'int8',
                'x': 'float32',
                'y': 'float32',
                'z': 'float32',
                'label': 'category'
            }
        )
        
        # 프레임 순서 유지
        df = df.sort_values(by=['frame_num']).reset_index(drop=True)
        
        # 프레임별로 그룹화
        grouped = df.groupby('frame_num')
        
        sequences = []
        labels = []
        
        current_sequence = []
        current_label = None
        
        for frame_num, group in grouped:
            # 프레임의 라벨 추출 (모든 키포인트가 같은 라벨을 가정)
            label = group['label'].iloc[0]
            
            # 시퀀스의 현재 라벨이 없거나 변경된 경우 초기화
            if current_label is None:
                current_label = label
            elif label != current_label:
                current_sequence = []
                current_label = label
            
            # 각 프레임의 키포인트 데이터 생성
            frame_data = {
                'pose': np.zeros((33, 3), dtype=np.float32),       # 33개의 포즈 키포인트
                'left_hand': np.zeros((21, 3), dtype=np.float32),  # 21개의 왼손 키포인트
                'right_hand': np.zeros((21, 3), dtype=np.float32)  # 21개의 오른손 키포인트
            }
            
            for _, row in group.iterrows():
                landmark_type = row['landmark_type']
                index = row['index']
                coords = [row['x'], row['y'], row['z']]
                
                if landmark_type in frame_data:
                    frame_data[landmark_type][index] = coords
            
            # 포즈, 왼손, 오른손 키포인트를 하나의 벡터로 결합
            concatenated_data = np.concatenate([
                frame_data['pose'],
                frame_data['left_hand'],
                frame_data['right_hand']
            ], axis=0)  # (75, 3)
            
            current_sequence.append(concatenated_data)
            
            # 시퀀스가 완성되면 리스트에 추가
            if len(current_sequence) == sequence_length:
                sequences.append(np.array(current_sequence))  # (SEQUENCE_LENGTH, 75, 3)
                labels.append(current_label)
                current_sequence = []
        
        # 데이터프레임 삭제 및 메모리 회수
        del df
        gc.collect()
        
        return sequences, labels
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [], []

# -----------------------------
# 2. 데이터 로딩 및 시퀀스 생성 (병렬 처리 적용)
# -----------------------------

def load_and_process_data(data_folder, sequence_length=30, max_workers=None):
    """
    지정된 폴더 내의 모든 CSV 파일을 병렬로 처리하여 시퀀스를 생성합니다.

    Parameters:
    - data_folder (str): CSV 파일들이 저장된 폴더의 경로.
    - sequence_length (int): 시퀀스당 프레임 수.
    - max_workers (int or None): 동시에 실행할 최대 작업자 수. 기본값은 None으로, 시스템의 CPU 수에 따름.

    Returns:
    - all_sequences (list of np.ndarray): 모든 시퀀스 리스트.
    - all_labels (list): 모든 시퀀스에 해당하는 라벨 리스트.
    """
    all_sequences = []
    all_labels = []
    
    # CSV 파일 경로 리스트 생성
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    # 병렬로 CSV 파일 처리
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # tqdm으로 진행 상태 표시
        results = list(tqdm(executor.map(process_csv, csv_files, [sequence_length]*len(csv_files)), 
                            total=len(csv_files), desc="CSV 파일 처리 중"))
    
    # 결과 통합
    for sequences, labels in results:
        all_sequences.extend(sequences)
        all_labels.extend(labels)
    
    return all_sequences, all_labels

# -----------------------------
# 3. 메인 실행
# -----------------------------

if __name__ == "__main__":
    # GPU 메모리 동적 할당 설정 (필요 시)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # GPU 메모리 증가 옵션 설정
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # 사용 가능한 GPU 목록 출력
            print("사용 가능한 GPU 목록:")
            for i, gpu in enumerate(gpus):
                print(f"GPU 번호 {i}: {gpu.name}")
        except RuntimeError as e:
            print(e)
    else:
        print("사용 가능한 GPU가 없습니다. CPU를 사용합니다.")
    
    # 데이터 폴더 경로 (실제 경로로 수정하세요)
    data_folder = 'data/labeled_keypoints/'
    
    # 시퀀스 생성 파라미터
    SEQUENCE_LENGTH = 30  # 시퀀스 당 프레임 수 (데이터에 맞게 조정)
    
    # 병렬 처리 최대 작업자 수 (기본적으로 시스템 CPU 수)
    max_workers = multiprocessing.cpu_count()  # 시스템의 CPU 코어 수 사용
    
    print(f"시스템의 CPU 코어 수: {max_workers}")
    
    # 데이터 로딩 및 시퀀스 생성
    print("데이터 로딩 및 시퀀스 생성을 시작합니다...")
    all_sequences, all_labels = load_and_process_data(data_folder, SEQUENCE_LENGTH, max_workers)
    
    print(f"총 시퀀스 수: {len(all_sequences)}")
    
    # -----------------------------
    # 4. 라벨 인코딩
    # -----------------------------
    
    # 라벨을 원-핫 인코딩
    print("라벨을 원-핫 인코딩 중...")
    y = pd.get_dummies(all_labels).values  # (num_sequences, num_classes)
    label_names = pd.get_dummies(all_labels).columns.tolist()  # 클래스 이름 리스트
    
    # -----------------------------
    # 5. 데이터 분할 (훈련/테스트)
    # -----------------------------
    
    print("데이터를 훈련 세트와 테스트 세트로 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(
        all_sequences, y, test_size=0.2,  random_state=42, stratify=y
    )
    
    print(f"훈련 샘플 수: {len(X_train)}, 테스트 샘플 수: {len(X_test)}")
    
    # -----------------------------
    # 6. 클래스 불균형 처리 (클래스 가중치)
    # -----------------------------
    
    print("클래스 가중치를 계산 중...")
    # 훈련 라벨을 정수로 변환
    y_train_integers = np.argmax(y_train, axis=1)
    
    # 클래스 가중치 계산
    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_integers),
        y=y_train_integers
    )
    
    # 클래스 가중치 딕셔너리 생성
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights_values)}
    print(f"클래스 가중치: {class_weights_dict}")
    
    # -----------------------------
    # 7. 데이터 정규화
    # -----------------------------
    
    print("데이터를 정규화 중...")
    # 훈련 및 테스트 데이터를 NumPy 배열으로 변환
    X_train = np.array(X_train)  # (num_train_samples, SEQUENCE_LENGTH, 75, 3)
    X_test = np.array(X_test)    # (num_test_samples, SEQUENCE_LENGTH, 75, 3)
    
    # 데이터 정규화를 위해 평탄화
    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]
    
    X_train_flat = X_train.reshape(num_train_samples * SEQUENCE_LENGTH, -1)  # (num_train_samples * SEQUENCE_LENGTH, 225)
    X_test_flat = X_test.reshape(num_test_samples * SEQUENCE_LENGTH, -1)     # (num_test_samples * SEQUENCE_LENGTH, 225)
    
    # 훈련 데이터의 평균과 표준편차 계산
    mean = X_train_flat.mean(axis=0)
    std = X_train_flat.std(axis=0) + 1e-8  # 0으로 나누는 것을 방지하기 위해 작은 값 추가
    
    # 정규화 적용
    X_train_flat = (X_train_flat - mean) / std
    X_test_flat = (X_test_flat - mean) / std
    
    # 원래 형태로 재변형
    X_train = X_train_flat.reshape(num_train_samples, SEQUENCE_LENGTH, -1)  # (num_train_samples, SEQUENCE_LENGTH, 225)
    X_test = X_test_flat.reshape(num_test_samples, SEQUENCE_LENGTH, -1)     # (num_test_samples, SEQUENCE_LENGTH, 225)
    
    # 메모리 확보
    del X_train_flat, X_test_flat, all_sequences, all_labels
    gc.collect()
    
    # -----------------------------
    # 8. GRU 모델 구축
    # -----------------------------
    
    print("GRU 모델을 구축 중...")
    model = Sequential()
    
    # Masking 레이어 (패딩이 있는 경우 무시)
    model.add(Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, X_train.shape[2])))
    
    # 첫 번째 GRU 레이어
    model.add(GRU(128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # 두 번째 GRU 레이어
    model.add(GRU(128, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # 출력 레이어 (소프트맥스 활성화)
    model.add(Dense(y.shape[1], activation='softmax'))
    
    # 모델 컴파일
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 모델 요약 출력
    model.summary()
    
    # -----------------------------
    # 9. 콜백 설정
    # -----------------------------
    
    print("콜백을 설정 중...")
    
    callbacks = [
        ModelCheckpoint('bestGRU_model.keras', monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]
    
    # -----------------------------
    # 10. 모델 학습
    # -----------------------------
    
    print("모델을 학습 중입니다...")
    batch_size = 32
    epochs = 100  # 필요에 따라 조정
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # -----------------------------
    # 11. 모델 평가
    # -----------------------------
    
    print("베스트 모델을 로드 중...")
    # 학습된 베스트 모델 불러오기
    best_model = load_model('bestGRU_model.keras')
    print("베스트 모델이 성공적으로 로드되었습니다.")
    
    print("테스트 세트로 모델을 평가 중...")
    # 테스트 데이터로 평가
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 정확도 (베스트 모델): {test_accuracy * 100:.2f}%")
    
    # 예측 수행
    print("예측을 수행 중...")
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # -----------------------------
    # 12. 결과 시각화
    # -----------------------------
    print("결과를 시각화 중...")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import seaborn as sns

    # NanumGothic 폰트 경로 확인
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    # 폰트 설정
    font_prop = font_manager.FontProperties(fname=font_path)

    # 음수 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

    # a. 혼동 행렬 (정규화 적용)
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes, normalize='true')  # 정규화 적용

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names)

    # 제목과 축 라벨에 폰트 적용
    plt.title('정규화된 혼동 행렬 (Normalized Confusion Matrix)', fontproperties=font_prop, fontsize=16)
    plt.xlabel('예측 라벨 (Predicted Label)', fontproperties=font_prop , fontsize=14)
    plt.ylabel('실제 라벨 (True Label)', fontproperties=font_prop, fontsize=14)

    # x축 틱 라벨에 폰트 적용
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop, rotation=45)

    # y축 틱 라벨에 폰트 적용
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_prop, rotation=0)

    plt.tight_layout()
    plt.show()

    # b. 분류 보고서
    print("분류 보고서를 출력합니다:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=label_names))

    # c. 학습 기록 시각화
    print("학습 기록을 시각화합니다...")
    plt.figure(figsize=(14, 6))

    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='훈련 정확도 (Train Accuracy)', color='blue')
    plt.plot(history.history['val_accuracy'], label='검증 정확도 (Validation Accuracy)', color='orange')
    plt.title('훈련 및 검증 정확도', fontproperties=font_prop, fontsize=16)
    plt.xlabel('에포크 (Epochs)', fontproperties=font_prop, fontsize=14)
    plt.ylabel('정확도 (Accuracy)', fontproperties=font_prop, fontsize=14)
    plt.legend(prop=font_prop)
    plt.grid(True)

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='훈련 손실 (Train Loss)', color='blue')
    plt.plot(history.history['val_loss'], label='검증 손실 (Validation Loss)', color='orange')
    plt.title('훈련 및 검증 손실', fontproperties=font_prop, fontsize=16)
    plt.xlabel('에포크 (Epochs)', fontproperties=font_prop, fontsize=14)
    plt.ylabel('손실 (Loss)', fontproperties=font_prop, fontsize=14)
    plt.legend(prop=font_prop)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("모든 작업이 완료되었습니다.")

