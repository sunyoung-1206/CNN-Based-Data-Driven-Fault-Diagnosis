import os
import numpy as np
import pandas as pd
import openpyxl
from PIL import Image
import random
from glob import glob

# 1. 폴더 및 저장 경로 설정
DATA_DIR = "./test2_data"  # 원본 데이터 파일 위치
OUTPUT_DIR = "./test2_generated_data"
TRAIN_IMAGE_DIR = os.path.join(OUTPUT_DIR, "train_images")
TEST_IMAGE_DIR = os.path.join(OUTPUT_DIR, "test_images")

# 폴더 생성 (권한 확인을 위한 예외 처리 추가)
try:
    os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
    os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
    print(f"✅ 폴더 생성 성공: {TRAIN_IMAGE_DIR}, {TEST_IMAGE_DIR}")
except Exception as e:
    print(f"❌ 폴더 생성 실패: {str(e)}")

# 2. 라벨 매핑 (BF, OF, IF, 정상 상태)
label_mapping = {
    "IR_1": 0, "IR_2": 0, "IR_3": 0, "IR_4": 0, "IR_5": 0,
    "IW_1": 1, "IW_2": 1, "IW_3": 1, "IW_4": 1, "IW_5": 1,
    "NO_1": 2, "NO_2": 2, "NO_3": 2, "NO_4": 2, "NO_5": 2,
    "OR_1": 3, "OR_2": 3, "OR_3": 3, "OR_4": 3, "OR_5": 3,
    "BR_1": 4, "BR_2": 4, "BR_3": 4, "BR_4": 4, "BR_5": 4
}

# 3. 파일 로드 및 디버깅 정보 추가
csv_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
print(f"📁 발견된 CSV 파일 수: {len(csv_files)}")
if len(csv_files) == 0:
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인하세요: {os.path.join(DATA_DIR, '*.csv')}")
    print(f"  현재 작업 디렉토리: {os.getcwd()}")

# 라벨 매핑 디버깅
unmapped_files = []
for file_path in csv_files:
    file_name = os.path.basename(file_path).replace(".csv", "")
    if file_name not in label_mapping:
        unmapped_files.append(file_name)
if unmapped_files:
    print(f"⚠ 라벨 매핑에 없는 파일들: {unmapped_files}")

# 데이터 포인트 개수 확인
file_data_points = {}
for file_path in csv_files:
    try:
        df = pd.read_csv(file_path, header=None)
        data = df.apply(pd.to_numeric, errors='coerce').to_numpy().ravel()
        data = data[~np.isnan(data)]
        file_name = os.path.basename(file_path)
        file_data_points[file_name] = len(data)
        print(f"📊 파일: {file_name}, 데이터 포인트 수: {len(data)}" +
              (", ✅ 충분함" if len(data) >= 4096 else f", ❌ 부족함 (최소 4096 필요)"))
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {file_path}, {str(e)}")

# 학습/테스트 데이터 저장용 리스트
train_data_list = []
test_data_list = []

# 실제 생성된 데이터 개수를 추적
valid_train_count = 0
valid_test_count = 0

# 각 파일별 생성 데이터 추적용 딕셔너리
file_train_counts = {}
file_test_counts = {}

# 4. 데이터 처리 및 이미지 변환 (4096개 선택 후 전처리) - 디버깅 정보 추가
def process_and_save_images(file_path, label, train_count=2000, test_count=400):
    global valid_train_count, valid_test_count

    file_name = os.path.basename(file_path)
    print(f"\n🔄 처리 중: {file_name} (라벨 {label})")

    """ 주어진 파일에서 데이터를 읽어 이미지로 변환하고 저장 """
    try:
        df = pd.read_csv(file_path, header=None)  # CSV 파일 로드
        df = df.apply(pd.to_numeric, errors='coerce')  # 문자열이 포함된 경우 숫자로 변환
        data = df.to_numpy().ravel()  # 1D numpy 배열 변환
        data = data[~np.isnan(data)]  # NaN 값 제거
        total_data_points = len(data)

        # 데이터가 부족하면 카운트하지 않고 종료
        if total_data_points < 4096:
            print(f"❌ 데이터 부족: {file_path} (데이터 개수: {total_data_points}), 최소 4096개 필요 - 스킵됨")
            return

        # 가능한 샘플 수 계산
        max_possible_samples = total_data_points - 4096 + 1

        # 실제 생성할 샘플 수 결정
        actual_train_count = min(train_count, max_possible_samples)
        actual_test_count = min(test_count, max_possible_samples)

        print(f"📈 가능한 최대 샘플 수: {max_possible_samples}")
        print(f"📋 생성 예정: 학습 데이터 {actual_train_count}개, 테스트 데이터 {actual_test_count}개")

        # 학습용 데이터 생성
        train_indices = random.sample(range(0, max_possible_samples), actual_train_count)
        train_success_count = 0

        for i in train_indices:
            segment = data[i:i + 4096]  # 4096개 데이터 선택

            # 최소값과 최대값 확인 (전처리 디버깅)
            min_val = np.min(segment)
            max_val = np.max(segment)

            # 최소값과 최대값이The assistant can create and refernce artifacts during conversations. Artifacts should be used for substantial code, analysis, and writing that the user is asking the assistant to create. the same인 경우 전처리 불가
            if min_val == max_val:
                print(f"⚠ 전처리 불가: 데이터 세그먼트의 최소값과 최대값이 같음 ({min_val})")
                continue

            # **전처리 공식 적용**
            segment = np.round((segment - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            # 64x64로 변환
            segment = segment.reshape(64, 64)
            img_path = os.path.join(TRAIN_IMAGE_DIR, f"train_{label}_{valid_train_count}.png")

            # 파일 저장 및 확인
            try:
                Image.fromarray(segment).convert("L").save(img_path)
                train_data_list.append([f"train_{label}_{valid_train_count}.png", label])
                valid_train_count += 1  # 유효한 학습 데이터 증가
                train_success_count += 1
            except Exception as e:
                print(f"⚠ 이미지 저장 오류 (학습 데이터): {img_path}, {str(e)}")
                continue  # 저장 실패 시 카운트하지 않음

        # 테스트용 데이터 생성
        test_indices = random.sample(range(0, max_possible_samples), actual_test_count)
        test_success_count = 0

        for i in test_indices:
            segment = data[i:i + 4096]  # 4096개 데이터 선택

            # 최소값과 최대값 확인 (전처리 디버깅)
            min_val = np.min(segment)
            max_val = np.max(segment)

            # 최소값과 최대값이 같은 경우 전처리 불가
            if min_val == max_val:
                print(f"⚠ 전처리 불가: 데이터 세그먼트의 최소값과 최대값이 같음 ({min_val})")
                continue

            # **전처리 공식 적용**
            segment = np.round((segment - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            # 64x64로 변환
            segment = segment.reshape(64, 64)
            img_path = os.path.join(TEST_IMAGE_DIR, f"test_{label}_{valid_test_count}.png")

            # 파일 저장 및 확인
            try:
                Image.fromarray(segment).convert("L").save(img_path)
                test_data_list.append([f"test_{label}_{valid_test_count}.png", label])
                valid_test_count += 1  # 유효한 테스트 데이터 증가
                test_success_count += 1
            except Exception as e:
                print(f"⚠ 이미지 저장 오류 (테스트 데이터): {img_path}, {str(e)}")
                continue  # 저장 실패 시 카운트하지 않음

        # 현재 파일 처리 결과 기록
        file_train_counts[file_name] = train_success_count
        file_test_counts[file_name] = test_success_count

        print(f"✅ {file_name} 처리 완료: 학습 데이터 {train_success_count}개, 테스트 데이터 {test_success_count}개 생성됨")

    except Exception as e:
        print(f"❌ 파일 처리 중 오류 발생: {file_path}, {str(e)}")

# 5. 모든 파일 처리
processed_file_count = 0
for file_path in csv_files:
    file_name = os.path.basename(file_path).replace(".csv", "")
    # 라벨 매핑
    if file_name in label_mapping:
        label = label_mapping[file_name]
        process_and_save_images(file_path, label)
        processed_file_count += 1
    else:
        print(f"⚠ 라벨 매핑 없음 (건너뜀): {file_name}")

print(f"\n📑 처리된 파일 수: {processed_file_count}/{len(csv_files)}")

# 6. 엑셀 파일 생성 (헤더 없이 저장)
def save_to_excel(data_list, file_name):
    """ 데이터 리스트를 엑셀 파일로 저장 (헤더 제거) """
    try:
        df = pd.DataFrame(data_list)
        output_path = os.path.join(OUTPUT_DIR, file_name)
        df.to_excel(output_path, index=False, header=False)  # 헤더 제거
        print(f"✅ 엑셀 파일 저장 성공: {output_path} ({len(data_list)}개 항목)")
    except Exception as e:
        print(f"❌ 엑셀 파일 저장 실패: {file_name}, {str(e)}")

save_to_excel(train_data_list, "trainingImageList.xlsx")
save_to_excel(test_data_list, "validationImageList.xlsx")

# 7. 각 파일별 처리 결과 요약
print("\n📊 파일별 처리 결과 요약:")
print("=" * 50)
print(f"{'파일명':<30} {'학습 데이터':<12} {'테스트 데이터':<12}")
print("-" * 50)
for file_name in sorted(set(list(file_train_counts.keys()) + list(file_test_counts.keys()))):
    train_count = file_train_counts.get(file_name, 0)
    test_count = file_test_counts.get(file_name, 0)
    print(f"{file_name:<30} {train_count:<12} {test_count:<12}")
print("=" * 50)

# 8. 총 처리 결과
print(f"\n✅ 데이터셋 생성 완료!")
print(f"📊 총 생성된 데이터: 학습 데이터 {valid_train_count}개 / 테스트 데이터 {valid_test_count}개")
print(f"📂 학습 이미지 폴더: {TRAIN_IMAGE_DIR}")
print(f"📂 테스트 이미지 폴더: {TEST_IMAGE_DIR}")
print(f"📄 학습 데이터 목록: {os.path.join(OUTPUT_DIR, 'trainingImageList.xlsx')}")
print(f"📄 테스트 데이터 목록: {os.path.join(OUTPUT_DIR, 'validationImageList.xlsx')}")
