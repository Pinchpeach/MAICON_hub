import nbformat, os, shutil
from nbclient import NotebookClient
d = "problem05"; shutil.copy(f"{d}/work.ipynb", f"{d}/versions/notebook_v2_softbox.ipynb")
nb = nbformat.read(f"{d}/versions/notebook_v1_box.ipynb", as_version=4)            # 단순한 v1 구조에서 출발
i_md = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and "상자 규칙" in c.source); i_code = i_md + 1
nb.cells[i_md].source = """## 풀이: 네 기준을 모두 통과하면 fit (상자 규칙) + status 별 회복일수 중앙값
단순한 구조를 유지하고, 80행 데이터의 특성(작은 표본, 기록 단위, 소수의 특이값)에 맞춰 과하게 맞추지 않는다.
- **status**: fit 10명은 모두 `pushup >= 38`, `run_time <= 9.6`, `sleep_quality >= 5.2`, `stress_level <= 5.8` 안에 있고 이 상자 안에 unfit 은 0명이다. 선형 경계(로지스틱, 정확도 0.9475)보다 이 규칙(0.967)이 맞는 형태다. 임계값은 fit 의 극값이지만 특이값에 끌린 것이 아니다: 다음 fit 과의 간격이 run_time 0.1, stress_level 0.3, pushup 0(2명이 같은 값)으로 작고, sleep_quality 만 0.6 이다. 모두 기록 단위(pushup 정수, 나머지 0.1)에 맞는 값이다.
- **회복일수**: 정수 일수이고 특이값이 있다(fit 의 8일은 10명 중 1명, unfit 의 16일은 70명 중 1명; 각각 평균에서 2 표준편차 밖). 평균(5.50 / 10.07)은 이 값들에 끌리므로 **status 별 중앙값(fit 5일, unfit 10일)**을 쓴다. 단위도 정수 일수와 맞는다.
  - 5-Fold x 20 에서 평균보다 높은 점수(+0.10, 20회 모두). 10% 절사평균은 +0.01, 윈저화 평균은 +0.01 로 차이가 없어, 가장 단순한 중앙값을 택했다.
- **제외한 복잡한 장치**: 임계값 불확실 구간의 확률 가중(SoftBox, 테스트 한 행 S035 용)은 "구간 안에서 임계값이 고르게 분포한다"는 가정에 기대고 구조가 복잡해 뺐다. S035(run_time 10.4, 기준 하나만 미달)는 규칙대로 unfit 이다.
- **검증**: Stratified 5-Fold x 20. 점수 = 50 x 정확도 + 50 / (1 + 회복 RMSE / 2.46475151)."""
src = nb.cells[i_code].source
src = src.replace("oof[b] = np.where(pb, rec[a][is_fit[a]].mean(), rec[a][~is_fit[a]].mean())", "oof[b] = np.where(pb, np.median(rec[a][is_fit[a]]), np.median(rec[a][~is_fit[a]]))")
src = src.replace("pred_fit = box.predict(test[features]); mean_fit, mean_unfit = rec[is_fit].mean(), rec[~is_fit].mean()", "pred_fit = box.predict(test[features]); mean_fit, mean_unfit = np.median(rec[is_fit]), np.median(rec[~is_fit])")
src = src.replace('print("회복일 평균: fit', 'print("회복일 중앙값: fit').replace("LogisticRegression C=1", "LogisticRegression C=1")
assert "np.median(rec[is_fit])" in src and "np.median(rec[a][is_fit[a]])" in src, "패치 실패"
nb.cells[i_code].source = src
for c in nb.cells:
    if c.cell_type == "code": c.outputs = []; c.execution_count = None
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": d}}).execute()
nbformat.write(nb, f"{d}/work.ipynb"); shutil.copy(f"{d}/submission.csv", f"{d}/versions/v3_box_median_main.csv"); print("done")
