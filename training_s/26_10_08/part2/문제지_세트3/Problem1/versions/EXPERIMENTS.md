# 세트3 Problem1 (카드 이상거래, AP) 실험 기록
평가: 5-Fold Stratified OOF AP (사기 358/70,000건, 노이즈 ±0.01 가능). Time 분포는 train/test 동일(랜덤 분할).

| 구성 | OOF AP |
|---|---|
| 로지스틱(표준화) | 0.8391 |
| HGB | 0.8817 |
| RF | 0.8757 |
| ExtraTrees | 0.8852 |
| HGB+RF+ET 순위평균 | 0.8887 |
| **LR+HGB+RF+ET 순위평균 (메인, v1)** | **0.8912** |
| (참고) +CatBoost 5개 | 0.8928, 부트스트랩 차이 95% [-0.0001, +0.0033] → 노이즈, 미채택 |

- 메인: `v1_rankavg_lr_hgb_rf_et_main.csv` = `submission.csv` (30000행, id,Class, 0~1 순위 점수). sklearn만 사용(서버 호환).
- 원본 노트북 백업: `notebook_orig.ipynb`. 노트북은 Restart & Run All 로 출력 저장.
- 미시도: 특징 파생(Amount 로그 등), 하이퍼파라미터 탐색 (기대 이득 작음, 천장은 라벨 수 한계).

## 정답 공개 후 채점 (평가 전용, 학습·튜닝에 미사용)
- 정답지_세트3.zip 의 solution.csv 로만 채점. solution_code 는 열지 않음.
- 실제 test AP **0.8306** (OOF 0.891 대비 낮음: test 양성 수가 적어 AP 변동이 큼, 분포 이동은 확인 안 됨)
