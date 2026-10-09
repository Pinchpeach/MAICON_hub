# 도's Pick! · 3번

대회 시간: **2026-10-09 08:50~10:20 (한국 시간)**. 시작 전에는 운영자 개방 승인이 필요합니다.

MAICON 모의고사 1회차 | 문제 3
군용 장비 운영 데이터 기반 고장 여부 예측

[문제 설명]
군용 정비 시설에서는 장비의 온도, 회전 속도, 토크 및 누적 마모 정보를 기록하여 고장을 조기에 발견하려 합니다. 주어진 운영 데이터를 분석하고, 각 장비 관측값에서 고장이 발생했는지 여부를 예측하는 이진 분류 모델을 구축하세요.

[제공 데이터]
- train.csv: 장비 종류(Type), 공기·공정 온도, 회전 속도, 토크, 마모 시간 및 고장 여부
- test.csv: 고장 여부 정답이 제거된 테스트 관측값
- sample_submission.csv: 제출 예시

[예측 대상]
- machine_failure = 0: 정상
- machine_failure = 1: 고장

[모델 및 구현 조건]
1. RandomForestClassifier를 기본 모델로 학습하고 성능을 평가하세요. 다른 모델을 이용한 성능 개선도 허용됩니다.
2. 범주형 변수 인코딩, 파생 변수 생성, 클래스 불균형 대응, 검증 및 임곗값 최적화 등을 자유롭게 활용할 수 있습니다.
3. 원본 데이터의 Machine failure 외에도 TWF, HDF, PWF, OSF, RNF는 고장 발생 여부 또는 유형을 알려주는 정답 관련 열입니다. 이 다섯 열은 학습 및 테스트 입력 특성으로 사용할 수 없습니다.

[평가]
고장(1)을 양성 클래스로 하는 F1-score를 사용합니다. 고장 사례는 정상 사례보다 매우 적으므로 단순 정확도보다 양성 클래스의 precision과 recall을 함께 고려하세요. 높을수록 좋습니다.

[제출 형식]
파일명: submission.csv
컬럼: id,machine_failure
- machine_failure는 0 또는 1의 정수여야 합니다.
- 테스트 관측값마다 정확히 한 행을 작성하고, sample_submission.csv의 id와 행 순서를 유지하세요.

[원본 데이터]
Predictive Maintenance Dataset (AI4I 2020)
https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

[태그]
#표형데이터 #전처리 #불균형분류 #고장예측 #RandomForest

## 이번 대회의 제공 구성
- 공개 학습: `data/train.csv` 8,000행. 정상 7,729행, 고장 271행
- 최종 평가: `data/test.csv` 2,000행. 정답은 공개하지 않습니다.
- 고장 여부를 층화해 80:20으로 분할했습니다. 학습·평가 양쪽에 두 클래스가 있습니다.
- `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`는 입력에서 제거했습니다.

| 컬럼 | 의미 |
|---|---|
| id | 원본 UDI 기준 고유 식별자 |
| type | 장비 종류: L/M/H |
| air_temperature_k | 공기 온도(K) |
| process_temperature_k | 공정 온도(K) |
| rotational_speed_rpm | 회전 속도(rpm) |
| torque_nm | 토크(Nm) |
| tool_wear_min | 누적 마모 시간(min) |
| machine_failure | 학습에만 있는 정답. 0 정상 / 1 고장 |

## 공통 제출 검증
- 최종 파일: `/workspace/problem03/submission.csv`
- 정확한 컬럼 순서: `id,machine_failure`
- sample_submission의 모든 ID를 정확히 한 번씩, 같은 순서로 제출합니다. 누락·추가·중복·순서 변경은 오류입니다.
- UTF-8 또는 UTF-8 BOM CSV, 쉼표 구분, 헤더 포함. 빈 필드·NaN·무한대·깨진 CSV는 오류입니다.
- 1번 수량은 0 이상의 정수, 2번은 지정된 5개 문자열, 3번은 문자열 `0` 또는 `1` 형태의 정수 클래스입니다.
- 유효하지 않은 제출은 해당 문제 0점·ERROR로 처리합니다.
- 문제 점수는 **100 × raw 평가 지표**입니다. 정답과 정확히 같으면 100점이며 세 문제 합계는 300점입니다.
- 제공 데이터는 읽기 전용이며, 가공 결과·모델·제출은 작업공간에 저장합니다.
