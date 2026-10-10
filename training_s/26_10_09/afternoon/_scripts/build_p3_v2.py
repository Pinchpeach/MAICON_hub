import nbformat, os, shutil
from nbclient import NotebookClient
d = "problem03"; nb = nbformat.read(f"{d}/work.ipynb", as_version=4); shutil.copy(f"{d}/work.ipynb", f"{d}/versions/notebook_v1_blend.ipynb")
i_md = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and "풀이" in c.source); i_code = i_md + 1
nb.cells[i_md].source += """
- **v2 (가중 평균 + 계수 반올림)**: 절편을 빼고 추정하면 계수 합이 0.991 로 1 에 가깝고, 합을 1 로 고정한 뒤 절편을 넣으면 -0.07 로 0 에 가깝습니다. 그래서 종합 만족도를 **네 점수의 가중 평균(계수 합 1, 절편 없음)**으로 보고, 추정한 가중치를 0.05 단위로 반올림합니다. 전체 학습에서는 0.289 / 0.251 / 0.244 / 0.216 이 **0.30 / 0.25 / 0.25 / 0.20** 이 됩니다.
- 같은 검증에서 RMSE 0.4585 -> 0.4538 (20회 중 15회에서 낮음, 점수 69.78 -> 70.00). 차이가 작고, 반올림 단위는 0.05 와 0.1 두 가지 중 좋은 쪽을 골랐습니다(0.1 은 0.515 로 나쁨). 잡음 하한(약 0.446)에 가까워 더 줄이기 어렵습니다.
- 서로 다른 모델 23종 비교: 선형 계열 10종(Ridge, Lasso, ElasticNet, BayesianRidge, Huber, Theil-Sen, PLS, 선형 SVR, 가우시안 프로세스 포함)이 0.463~0.465 로 몰려 있고, 2차 다항 0.473, RBF SVR 0.491, 부스팅 0.57~0.65, kNN 0.61~0.63, 신경망 0.685, RandomForest 0.696 으로 유연할수록 나빴습니다."""
src = nb.cells[i_code].source
add = '''
class WeightedMean:
    """v2 메인: 계수 합 1, 절편 없음(가중 평균)으로 추정한 뒤 가중치를 grid 단위로 반올림"""
    def __init__(self, grid=0.05): self.grid = grid
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); w3 = np.linalg.lstsq(X[:, :3] - X[:, [3]], y - X[:, 3], rcond=None)[0]
        w = np.r_[w3, 1 - w3.sum()]; w = np.round(w / self.grid) * self.grid; w[np.argmax(w)] += 1 - w.sum(); self.w = w; return self
    def predict(self, X): return np.asarray(X, float) @ self.w
'''
marker = "features = [c for c in test.columns"; assert marker in src and "WeightedMean" not in src
src = src.replace(marker, add.lstrip("\n") + "\n" + marker)
src = src.replace('for n, v in (("선형", a), ("평균 한 변수 선형", b), ("두 모델 평균 (메인)", c)):', 'd = cv_rmse(X, y, WeightedMean)\nfor n, v in (("선형", a), ("평균 한 변수 선형", b), ("두 모델 평균 (v1)", c), ("가중 평균 0.05 반올림 (메인 v2)", d)):')
src = src.replace('print("메인이 선형보다 낮은 반복:", int((c < a).sum()), "/", R)', 'print("v1 이 선형보다 낮은 반복:", int((c < a).sum()), "/", R, "| v2 가 v1 보다 낮은 반복:", int((d < c).sum()), "/", R)')
old = src[src.index("model = Blend().fit(X, y)"):src.index("submission = pd.DataFrame")]
src = src.replace(old, 'model = WeightedMean().fit(X, y)\nprint("가중치:", dict(zip(features, model.w.round(3))), "| 합", round(float(model.w.sum()), 3), "| 훈련 잔차 표준편차 %.4f" % (y - model.predict(X)).std())\n')
assert "WeightedMean().fit" in src and "메인 v2" in src
nb.cells[i_code].source = src
for c in nb.cells:
    if c.cell_type == "code": c.outputs = []; c.execution_count = None
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": d}}).execute()
nbformat.write(nb, f"{d}/work.ipynb"); shutil.copy(f"{d}/submission.csv", f"{d}/versions/v2_weighted_mean_rounded_main.csv"); print("done")
