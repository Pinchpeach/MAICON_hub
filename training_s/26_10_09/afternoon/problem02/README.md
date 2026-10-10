# 2번 · 사진 속 인원 집계

원본 문제: **S11-2**. 대회: 2026-10-09 14:45~16:35 KST.

**COCO 보완 문제 안내:** 실제 사진은 일반 생활 장면이며 군 행사 사진이 아닙니다. `train_images`와 `train_labels.csv`를 참고해 `test_images`에 등장한 사람 수를 예측하세요. 정답은 COCO 2017 validation `person` instance annotation의 수이며 `iscrowd=1` 대상 이미지는 제외됩니다. 제출: `filename,person_count`; 점수: MAE(낮을수록 좋음), 완전 일치율(높을수록 좋음). `unit_name`은 실제 부대명이 아닙니다.


## 제출·채점

- 저장 경로: `/workspace/problem02/submission.csv`
- 컬럼 순서: `filename,person_count`
- sample_submission.csv의 모든 ID를 같은 순서로 정확히 한 번 제출합니다.
- 점수: **100 / (1 + Count MAE)**, 문제당 100점. 5문제 합계 500점.
- 원본 패키지에는 종합점수가 없어, 원래 지표를 유지하는 단조 변환으로 통합 점수를 정했습니다. 회귀의 scale은 공개 학습 정답의 모집단 표준편차입니다.
- NaN·무한대·빈 필드·중복/누락 ID·잘못된 컬럼은 0점·ERROR입니다. 수량은 0 이상 정수, status는 fit/unfit, 회복일수는 0 이상 유한수입니다.
- 제공 데이터는 읽기 전용입니다. 결과 파일은 작업공간에 저장하세요.
- 종료 후에도 사후 제출이 가능하며 공식 종료 점수는 별도로 보존합니다.

## 이미지 안내

군사 사진 대신 COCO 2017 생활 장면을 사용합니다. 두 이미지 문제는 서로 다른 60장씩이며 각각 학습 30장·평가 30장입니다. 사람은 COCO person, 차량은 car+bus+truck이며 자전거·오토바이는 제외합니다. 정답은 모델 추론값이 아닌 공개 객체 주석입니다. 원본 주석에서 평가 답을 조회하지 마세요. 이미지 출처·라이선스는 data/image_credits.csv를 확인하세요.
