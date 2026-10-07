# 24 Problem#2 실험 기록 (5-Fold KFold(shuffle, seed=42), RMSE, 낮을수록 좋음)
단일 모델 CV RMSE: hgb 59905.3 | rf 59916.8 | et 60415.7 | hgb_log 60293.2 | hgb2 59560.7 | gbr_huber 59651.6 | ridge 70383.2 | (extra 교호작용 특징은 효과 없음)

| 버전 | 구성 | CV RMSE | 결정 |
|---|---|---|---|
| v1 | hgb+rf+et | 59103.5 | 보존(versions/v1_hgb_rf_et.csv) |
| v2 | v1 + 교호작용 특징 | 59049.5 | 차이 미미, 기각 |
| v3 | v1 + gbr_huber | 58910.5 | |
| v4 | hgb2+rf+et | 59117.5 | |
| v5 | hgb+hgb_log+rf+et | 58921.2 | |
| v6 | hgb+hgb_log+hgb2+rf+et+gbr_huber | **58809.1** | **채택(메인)**. 개선 폭 0.5%로 작음, 노이즈 가능성 있음 |

확인용(채택 판단에는 미사용): test_with_labels RMSE v1 57808 -> v6 57662. 비교는 같은 fold에서 수행.
