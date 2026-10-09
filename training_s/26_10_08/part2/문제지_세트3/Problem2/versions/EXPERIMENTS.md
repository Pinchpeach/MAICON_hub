# 세트3 Problem2 (KLUE-YNAT 7분류, Macro F1)
5-Fold Stratified OOF Macro F1 (30,000건). 클래스 비율 11~18%로 비교적 균형.
| 구성 | OOF F1 |
|---|---|
| 글자NB 1-2 / 1-3 / 1-4 (α=0.1) | 0.812 / 0.823 / 0.823 |
| 글자 LR(C=20) | 0.832 (97초, 느림) |
| **글자 1-4 TF-IDF LinearSVC C=0.3 (메인 v1)** | **0.841** |
| SVC C 0.1/0.2/0.5 | 0.834/0.840/0.839 |
| 1-5gram / +단어 / balanced / char(wb 없이) | 0.841 / 0.836 / 0.841 / 0.834 |
| NB+SVC, NB+LR 순위 평균 | 0.829 / 0.826 (낮아서 기각) |
- 변형 간 차이는 노이즈 수준 → 13.11에 따라 중단. 메인 = `v1_char_svc_main.csv` = `submission.csv`.
- 천장 참고: 사전학습 BERT 계열이 이 데이터에서 더 높을 수 있으나 CPU 2코어·다운로드 허락 문제로 미시도(미측정).
- 베이스라인(단어 1,000개 NB)의 OOF 점수는 측정하지 않음. 원본 노트북은 `notebook_orig.ipynb`.

## 정답 공개 후 채점 (평가 전용, 학습·튜닝에 미사용)
- 정답지_세트3.zip 의 solution.csv 로만 채점. solution_code 는 열지 않음.
- 실제 test Macro F1 **0.8467** (OOF 0.841과 일치)
