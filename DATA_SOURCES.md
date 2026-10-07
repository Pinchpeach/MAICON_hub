# 데이터·가중치 출처 (저장소에 올리지 않은 파일)

용량 한계(GitHub 파일당 100MB)나 재현 가능성 때문에 아래 파일은 저장소에서 제외했습니다. 로컬에는 있습니다. 압축해서 올린 것은 "저장소에 포함" 표를 참고하세요.

## 저장소에서 제외 (출처만 기록)

| 경로 | 크기 | 출처 / 복원 방법 | 제외 사유 |
|---|---|---|---|
| `Kaggle/P1/dataset/` (train 3,422개 + test 10개 + `train.csv` + `sample_submission.csv`) | 약 644MB | Kaggle 대회 **Global Wheat Detection** (`global-wheat-detection`). Kaggle 계정에서 대회 규정에 동의한 뒤 `kagglehub.competition_download('global-wheat-detection', ...)` 또는 Kaggle 웹/CLI로 내려받는다. 노트북 `Kaggle/P1/global-wheat-detection.ipynb`가 이 흐름을 사용한다 | 이미지라 압축해도 636MB(시험 결과)로 100MB 한도 초과 |
| `MAICON_pre/25MAICON_기출/yolov8x.pt` | 136.9MB | Ultralytics 공식 사전학습 가중치(YOLOv8x). `YOLO("yolov8x.pt")` 호출 시 자동 다운로드(릴리스: github.com/ultralytics/assets) | 압축해도 126.6MB로 한도 초과 |
| `MAICON_pre/25MAICON_기출/yolov8x-cls.pt` | 115.1MB | Ultralytics 공식 사전학습 가중치(YOLOv8x-cls, ImageNet 분류). `YOLO("yolov8x-cls.pt")` 호출 시 자동 다운로드 | 압축해도 106.4MB로 한도 초과 |
| `training_s/**/yolov8s.pt`, `yolov8m-cls.pt` | 22MB / 33MB | 위와 같은 Ultralytics 공식 가중치. 노트북 실행 시 자동 다운로드 | 재다운로드 가능 |
| `training_s/25MAICON_기출/Problem#1/data/`, `Problem#2/data/` (압축 전 원본 폴더) | 약 64MB, 66MB | 아래 zip 과 같은 내용 | zip 으로 포함했으므로 원본 폴더는 중복 |
| `training_s/26_10_07/Problem3/data/` (압축 전 원본 폴더) | 약 38MB | 아래 zip 과 같은 내용 | zip 으로 포함했으므로 원본 폴더는 중복 |

## 저장소에 포함 (압축해서 올림)

| 경로 | 크기(압축 후) | 내용 / 출처 |
|---|---|---|
| `training_s/25MAICON_기출/problem_1_data.zip`, `problem_2_data.zip`, `problem_3_data.zip` | 67.0MB / 53.1MB / 0.04MB | 25년 MAICON 기출 문제 제공 데이터(전차·자동차 이미지, 군함·선박 이미지와 라벨, 스트레스 텍스트) |
| `training_s/26_10_07/Problem3/data.zip` | 9.8MB | 모의 대회 P3 데이터(`train.csv`, `test.csv`, `sample_submission.csv`). 문제 지문상 출처: Fashion-MNIST(Zalando Research, MIT License) 일부를 대회용으로 재구성 |
| `MAICON_pre/25MAICON_기출/runs/detect/{full,split}/weights.zip` | 각 41MB | 25년 P2 YOLOv8s 전이학습 가중치(best.pt, last.pt). 재학습 시 CPU에서 약 25~30분 |

## 저장소에 그대로 포함된 작은 데이터와 출처

| 경로 | 출처 |
|---|---|
| `training_s/24MAICON_기출/*/data` | 24년 MAICON 기출 문제 제공 데이터 |
| `training_s/25MAICON_기출/Problem#3/data` | 25년 MAICON 기출 P3 데이터 |
| `training_s/26_10_07/Problem1/data` | 모의 대회 P1. 문제 지문상 출처: ggplot2 diamonds 데이터셋을 대회용으로 재구성 |
| `training_s/26_10_07/Problem2/data` | 모의 대회 P2. 문제 지문상 출처: NSMC(Naver Sentiment Movie Corpus, CC0) 일부를 대회용으로 재구성 |

## 올리지 않은 파일 (비밀 정보)
- `Kaggle/P1/global-wheat-detection.ipynb`: Kaggle API 토큰이 평문으로 들어 있어 제외했습니다. 토큰을 제거한 뒤 올리세요.
