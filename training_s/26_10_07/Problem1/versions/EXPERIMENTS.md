# 26_10_07 Problem1 (다이아몬드 가격, RMSLE) 실험 기록
검증: 5-Fold KFold(shuffle, seed 42), log1p(price) 공간의 RMSE = RMSLE (낮을수록 좋음). 환산 점수 = 100 x (1 - RMSLE / 0.3938).

## 오차 분석 (v1 OOF)
- 오차 상위 1% 행이 전체 제곱오차의 20.6%, 상위 5%가 45.9%. 최악 행은 Fair 컷의 작은 돌인데 가격이 높은 경우 등 라벨 노이즈 성격.
- x/y/z 이상(0 또는 비정상 큰 값) 15행은 RMSLE 0.40 이지만 전체 SSE의 0.9%.
- depth/table 결측 행은 오차가 평균과 비슷(각 SSE 2.7%/3.5%).
- 큰 돌(carat>1.5)과 I1/SI2 에서 오차가 큼. test 행 중 train 과 특징이 동일한 행이 105개.

## 버전 비교
| 버전 | 구성 | CV RMSLE | 환산 점수 | 결정 |
|---|---|---|---|---|
| 정리만 한 HGB | 범주 정리, NaN native | 0.0896 | | 기각 |
| HGB + depth/z 복원 | | 0.0895 | | 효과 거의 없음 |
| HGB + 파생 특징 | log_carat, vol, 교호작용 등 | 0.0851 | | 파생 특징이 가장 큰 효과 |
| **v1 (메인)** | HGB + RF 평균 | **0.0833** (재측정 0.0831) | 약 78.8 | **채택(메인)**, 제출물 versions/v1_hgb_rf_feats.csv |
| v2 후보 | + x/y/z 모두 0 일 때 캐럿으로 추정 | HGB 0.0851 (변화 없음) | | 효과 없음 |
| v2 후보 | HGB L1(absolute_error) | 0.0869 | | 단독으로는 나쁨, 앙상블 다양성용 |
| v2 후보 | ExtraTrees(500) | 0.0871 | | 단독 약함 |
| v2 후보 | HGB+RF+ET 단순 평균 | 0.0831 | | v1 과 동일 수준 |
| v2 후보 | HGB+HGB_L1+RF+ET 단순 평균 | 0.0828 (MAE 0.0567) | 79.0 | 노이즈 수준 |
| v2 후보 | 위 4모델 검증 기반 NNLS 가중 평균(w: hgb .39, l1 .19, rf .11, et .31) | 0.0827 (가중치 자체를 교차검증) | 79.0 | 노이즈 수준 |
| v2 후보 | 위 4모델 Ridge(positive) 스태킹 | 0.0828 | | 노이즈 수준 |

결론: sklearn 계열 알고리즘/가중치 변경은 0.0833 -> 0.0827 로 사실상 천장(약 0.083). 차이(0.0006)는 fold 노이즈 수준이라 v1 유지.
CatBoost 비교는 설치 허락 후 진행 예정(사용자가 제시한 CatBoost + ExtraTrees 가중 평균 방향).

---
## CatBoost 비교 (사용자 제시 방향: CatBoost + ExtraTrees 검증 기반 가중 평균). catboost 1.2.10 설치 후 측정
설정: iterations 1500, lr 0.06, depth 8, l2 3, RMSE, seed 0. 동일 5-Fold. 학습 시간(CPU 16 스레드): cb_ord 약 11초/fold, cb_cat 약 80초/fold.

| 버전 | 구성 | CV RMSLE | MAE | 큰 오차(>0.3) | 비고 |
|---|---|---|---|---|---|
| cb_ord | CatBoost, 등급을 순서 숫자로 | 0.0823 | | | HGB 0.0851 보다 좋음 |
| cb_cat | CatBoost, 등급을 범주형으로 직접 | 0.0824 | | | cb_ord 와 동급이나 7배 느림 |
| cb_ord + et | 단순 평균 / NNLS(.73/.27) | 0.0821 / 0.0816 | 0.0566 | 0.59% | |
| cb_ord + cb_cat | 단순 평균 | 0.0819 | 0.0571 | 0.54% | |
| **v2 (메인)** | **cb_ord + cb_cat + et 단순 평균(1/3씩)** | **0.0814** (NNLS 0.0813, Ridge 0.0813) | 0.0563 | 0.55% | **채택**, 환산 점수 약 79.3 |
| | 6모델(+hgb, hgb_l1, rf) NNLS/Ridge | 0.0813 | | | v2 와 동일해 복잡도만 늘어 기각 |

- v1(0.0831) 대비 RMSLE 약 2.0% 개선(환산 약 +0.5점). 개선 폭은 작지만 CatBoost 단독으로 HGB 대비 3% 좋아졌고, 블렌드 가중치도 동일 비중에 가까워 단순 평균을 채택.
- 노트북에는 시간 절약을 위해 빠른 모델(cb_ord, et)의 CV 만 포함(약 1.5분). cb_cat 포함 전체 CV 는 위 표의 실험값. 노트북 전체 실행 시간 약 4분.
- 제출물: versions/v2_cb_ord_cb_cat_et.csv. v1 노트북은 versions/notebook_v1_hgb_rf.ipynb 로 보존.
- 주의: CatBoost 는 문제의 허용 목록(pandas, scikit-learn, LightGBM, XGBoost, OpenCV, PyTorch 등 일반 라이브러리)에 명시되어 있지 않음. 대회 환경에 없으면 `pip install catboost`(인터넷 가능) 필요. 사용 불가 시 v1(sklearn 만)이 대체안.

---
## 개선 후보 1(캐럿 경계 특징), 2(타깃 변환), 4(노이즈 행 제거) 비교 (CatBoost 순서 인코딩 기준, 동일 5-Fold)
2, 4 는 각 fold 의 학습 부분 안에서만 계산(검증 누수 없음). 4 는 학습 fold 내부 3-Fold 잔차 상위 1%(1,600행)를 제외하고 학습, 검증 fold 는 전체 행.

| 버전 | cb_ord CV RMSLE | HGB CV RMSLE |
|---|---|---|
| A 기준 | 0.0823 | 0.0851 |
| B +(1) 캐럿 경계 특징 | 0.0820 | 0.0846 |
| **C +(2) 타깃 변환(log1p(price)에서 a*log(carat)+b 제거 후 복원)** | **0.0816** | **0.0840** |
| D +(1)+(2) | 0.0816 | 0.0840 |
| E +(4) 노이즈 행 제외 | 0.0827 (악화) | |
| F +(1)+(2)+(4) | 0.0826 (악화) | 0.0846 |

- (2)가 가장 효과적(CatBoost -0.9%, HGB -1.3%). (1)은 소폭이고 (2)와 같이 써도 추가 이득 없음 -> 미채택. (4)는 오히려 악화 -> 기각(노이즈 행에도 test 와 공통된 신호가 있거나, 제외가 분포를 왜곡).
- ExtraTrees 는 타깃 변환 시 악화(0.0871 -> 0.0879) -> ET 는 변환 없이 사용.
- sklearn 만 쓰는 경우(v1'): hgb_off + rf_off 평균 0.0827 (v1 0.0831).

## v3 (메인): cb_off + cb_cat + et
| 블렌드(동일 비중 평균) | CV RMSLE |
|---|---|
| v2: cb_ord + cb_cat + et | 0.0814 |
| **v3: cb_off + cb_cat + et** | **0.0810** (환산 약 79.4) |
| cb_off + cb_ord + cb_cat + et | 0.0809 (4모델, 0.0001 차이라 미채택) |
| cb_off + et | 0.0817 |
| cb_off + hgb_off + et | 0.0812 |

- v3 구성: cb_off = 타깃 변환 CatBoost(순서 인코딩, 시드 3개), cb_cat = 범주형 직접 입력 CatBoost(변환 없음), et = ExtraTrees(변환 없음, 시드 2개). 로그 공간에서 1/3 씩 평균.
- 개선 폭: v2 대비 약 0.5%, v1 대비 약 2.5%. 소폭이며 fold 노이즈 범위에 가까움에 유의.
- 노트북 안의 CV 는 빠른 모델(cb_off 0.0816, et 0.0871, 평균 0.0817)만 재현. 3모델 블렌드 0.0810 은 실험값.
- 제출물: versions/v3_cb_off_cb_cat_et.csv (id 집합 일치, 중복/NaN/음수 없음). v2 노트북은 versions/notebook_v2_cb_cb_et.ipynb 로 보존.
- 이어서 실험하던 'cb_cat + 타깃 변환' 은 시간 절약을 위해 중단(미측정).
