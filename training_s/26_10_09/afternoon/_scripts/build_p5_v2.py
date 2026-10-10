import nbformat, os, shutil
from nbclient import NotebookClient
d = "problem05"; nb = nbformat.read(f"{d}/work.ipynb", as_version=4); shutil.copy(f"{d}/work.ipynb", f"{d}/versions/notebook_v1_box.ipynb")
i_md = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and "상자 규칙" in c.source); i_code = i_md + 1
nb.cells[i_md].source += """
- **v2 (임계값 불확실 구간의 회복일 확률 가중)**: 각 기준의 실제 임계값은 "fit 의 극값"과 "그 기준만 미달한 unfit 중 가장 가까운 값" 사이 어딘가에 있습니다(pushup 35~38, run_time 9.6~11.0, sleep_quality 4.0~5.2, stress_level 5.8~6.0). 이 구간에 걸린 행은 구간 안 위치에 비례해 fit 확률 p 를 주고, **status 는 p > 0.5 일 때만 fit, 회복일은 p x fit 평균 + (1-p) x unfit 평균**으로 예측합니다. 구간 밖의 행은 v1 과 같습니다.
- 테스트에서 이 구간에 걸린 행은 S035(run_time 10.4) 한 명이며 p = 0.43 → status unfit, 회복일 8.11일. 기대 점수 모의에서 p 가 0.43~0.5 일 때 약 +0.3~0.4점입니다. 교차검증에서는 차이가 없었습니다(+0.009).
- **상한**: status 를 모두 맞혀도 회복일 잡음(unfit 표준편차 2.05, fit 0.92) 때문에 기대 점수는 약 77.9점입니다. unfit 안에서 회복일을 설명하는 모델(Ridge, kNN, RF, 부스팅, 단일 변수 선형)은 모두 그룹 평균보다 같거나 나빴습니다."""
src = nb.cells[i_code].source
add = '''
class SoftBox(Box):
    """Box + 임계값 불확실 구간. lo = 그 기준만 미달한 unfit 중 가장 가까운 값. p_fit = 기준별 통과 확률의 곱"""
    def fit(self, X, y):
        super().fit(X, y); V = np.asarray(X, float) * SIGN; y = np.asarray(y, bool); self.lo = self.th.copy()
        for j in range(4):
            others = np.all(np.delete(V, j, 1) >= np.delete(self.th, j), axis=1); below = V[(~y) & others & (V[:, j] < self.th[j]), j]
            self.lo[j] = below.max() if len(below) else V[:, j].min() - 1e-9
        return self
    def proba(self, X):
        V = np.asarray(X, float) * SIGN; w = np.where(self.th > self.lo, self.th - self.lo, 1.0); return np.clip((V - self.lo) / w, 0, 1).prod(1)
'''
marker = "X = train[features]; is_fit"
assert marker in src and "SoftBox" not in src
src = src.replace(marker, add.lstrip("\n") + "\n" + marker)
old_tail = src[src.index("box = Box().fit(X, is_fit)"):]
new_tail = '''box = SoftBox().fit(X, is_fit)
print("임계값: pushup >= %.0f, run_time <= %.1f, sleep_quality >= %.1f, stress_level <= %.1f" % tuple(box.th * SIGN))
print("불확실 구간의 반대쪽 끝(가장 가까운 unfit): pushup %.0f, run_time %.1f, sleep_quality %.1f, stress_level %.1f" % tuple(box.lo * SIGN))
p_fit = box.proba(test[features]); pred_fit = p_fit > 0.5; mean_fit, mean_unfit = rec[is_fit].mean(), rec[~is_fit].mean()
submission = pd.DataFrame({"soldier_id": test.soldier_id, "status": np.where(pred_fit, "fit", "unfit"), "recovery_days": p_fit * mean_fit + (1 - p_fit) * mean_unfit})[sample.columns]
assert set(submission.status) <= {"fit", "unfit"} and (submission.recovery_days >= 0).all() and np.isfinite(submission.recovery_days).all()
submission.to_csv("submission.csv", index=False)
print("회복일 평균: fit %.2f, unfit %.2f | 테스트 fit 예측 %d/%d (train 비율 %.3f)" % (mean_fit, mean_unfit, pred_fit.sum(), len(test), is_fit.mean()))
print("구간에 걸린 테스트 행(0 < p < 1):"); print(test.assign(p_fit=p_fit.round(3), recovery_days=submission.recovery_days.round(2))[(p_fit > 0) & (p_fit < 1)].to_string(index=False))
submission.head()'''
src = src.replace(old_tail, new_tail); nb.cells[i_code].source = src
for c in nb.cells:
    if c.cell_type == "code": c.outputs = []; c.execution_count = None
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": d}}).execute()
nbformat.write(nb, f"{d}/work.ipynb"); shutil.copy(f"{d}/submission.csv", f"{d}/versions/v2_softbox_hedged_main.csv"); print("done")
