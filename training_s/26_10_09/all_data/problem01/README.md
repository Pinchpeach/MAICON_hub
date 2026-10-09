# 도's Pick! · 1번

대회 시간: **2026-10-09 08:50~10:20 (한국 시간)**. 시작 전에는 운영자 개방 승인이 필요합니다.

MAICON 모의고사 1회차 | 문제 1
군사 감시 이미지 기반 주요 차량 수량 예측

[문제 설명]
군사 감시 체계에서 확보한 이미지에는 전차, 군용 트럭, 기타 군용 차량, 민간 차량 등이 등장합니다. 주어진 학습 이미지와 객체 위치·종류 주석을 이용하여 이미지 속 주요 차량을 탐지하는 모델을 구축하세요. 테스트 이미지마다 지정된 4종의 객체가 각각 몇 대 있는지 예측해야 합니다.

[제공 데이터]
- train/images/: 학습 이미지
- train/labels/: YOLO 형식의 객체 탐지 정답(클래스 ID, 바운딩박스)
- test/images/: 정답이 공개되지 않은 테스트 이미지
- sample_submission.csv: 제출 예시
- data.yaml: 클래스 ID와 클래스 이름의 대응 관계

[예측 대상]
- military_tank: 전차 수
- military_truck: 군용 트럭 수
- military_vehicle: 기타 군용 차량 수
- civilian_vehicle: 민간 차량 수
※ 나머지 클래스는 이번 문제의 채점 대상이 아닙니다. 객체 1개는 가장 적절한 클래스 1개로만 셉니다.

[모델 및 구현 조건]
1. YOLOv8 계열의 사전학습 객체 탐지 모델을 사용하고, 제공된 학습 데이터로 전이학습을 수행하세요.
2. 입력 크기, 데이터 증강, 학습 횟수, confidence threshold, NMS 설정 등은 자유롭게 조정할 수 있습니다.
3. 테스트 정답 파일을 직접 읽거나 외부 정답을 가져오는 행위는 허용하지 않습니다.

[평가]
이미지별 '정확한 수량 일치율(Exact Count Accuracy)'을 사용합니다.
한 이미지에서 4개 클래스의 예측 수량이 모두 정답과 같을 때만 해당 이미지를 1점, 그렇지 않으면 0점으로 계산합니다. 최종 점수는 전체 테스트 이미지에서의 평균입니다. 높을수록 좋습니다.

[제출 형식]
파일명: submission.csv
컬럼: image_id,military_tank,military_truck,military_vehicle,civilian_vehicle
- image_id는 sample_submission.csv와 동일한 이미지 식별자를 사용하세요.
- 모든 수량은 0 이상의 정수여야 합니다.
- 테스트 이미지마다 정확히 한 행을 작성하고, sample_submission.csv의 행 순서를 유지하세요.

[원본 데이터]
Military Assets Dataset (12 Classes - Yolo8 Format)
https://www.kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format

[태그]
#이미지분석 #YOLOv8 #전이학습 #객체탐지 #수량예측

## 이번 대회의 제공 구성
- 공개 학습: `data/train/images/` 640장과 `data/train/labels/` YOLO 라벨
- 공개 검증: `data/val/images/` 160장과 `data/val/labels/` YOLO 라벨
- 최종 평가: `data/test/images/` 200장. 최종 평가 라벨은 제공하지 않습니다.
- 동일 이미지 바이트가 학습·검증·평가에 중복되지 않도록 선정했습니다.
- 원본의 비정상 라벨이 있는 이미지는 선정에서 제외했습니다.
- 원본 12개 클래스 중 지정 4개만 남기고 라벨을 다음과 같이 재매핑했습니다.

| 공개 class ID | 클래스 | 원본 class ID |
|---|---|---|
| 0 | military_tank | 2 |
| 1 | military_truck | 3 |
| 2 | military_vehicle | 4 |
| 3 | civilian_vehicle | 7 |

각 라벨 행은 `class_id x_center y_center width height`이며 좌표는 이미지 크기에 대한 0~1 정규화 값입니다. 빈 라벨 파일은 지정 4종 객체가 없는 이미지입니다. 다른 객체는 이번 학습 라벨과 수량 평가에서 제외합니다.
`data/data.yaml`에는 실제 경로와 4개 클래스의 대응이 있습니다. 데이터는 읽기 전용입니다. YOLO 학습 출력·가중치·캐시는 작업 폴더에 저장하세요.
공개 사전학습 가중치 `yolov8n.pt`를 미리 준비했습니다. Python의 `ultralytics`, CPU PyTorch도 설치되어 있습니다. 제공된 데이터로 전이학습 후 사용하세요.

## 공통 제출 검증
- 최종 파일: `/workspace/problem01/submission.csv`
- 정확한 컬럼 순서: `image_id,military_tank,military_truck,military_vehicle,civilian_vehicle`
- sample_submission의 모든 ID를 정확히 한 번씩, 같은 순서로 제출합니다. 누락·추가·중복·순서 변경은 오류입니다.
- UTF-8 또는 UTF-8 BOM CSV, 쉼표 구분, 헤더 포함. 빈 필드·NaN·무한대·깨진 CSV는 오류입니다.
- 1번 수량은 0 이상의 정수, 2번은 지정된 5개 문자열, 3번은 문자열 `0` 또는 `1` 형태의 정수 클래스입니다.
- 유효하지 않은 제출은 해당 문제 0점·ERROR로 처리합니다.
- 문제 점수는 **100 × raw 평가 지표**입니다. 정답과 정확히 같으면 100점이며 세 문제 합계는 300점입니다.
- 제공 데이터는 읽기 전용이며, 가공 결과·모델·제출은 작업공간에 저장합니다.
