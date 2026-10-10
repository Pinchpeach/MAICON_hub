import nbformat, os, shutil
from nbclient import NotebookClient
d = "problem01"; nb = nbformat.read(f"{d}/work.ipynb", as_version=4)
shutil.copy(f"{d}/work.ipynb", f"{d}/versions/notebook_v1_tobit.ipynb")
i_md = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and "Tobit" in c.source); i_code = i_md + 1
nb.cells[i_md].source += """
- **v2 (기울기 반올림)**: Tobit 으로 추정한 기울기를 정해진 격자(수면 1.0, 야간근무 0.5, 카페인 0.05)로 반올림한 뒤 절편과 잡음 크기만 다시 추정합니다. 전체 학습에서는 -6.08 / +4.42 / +0.101 이 **-6 / +4.5 / +0.10** 이 됩니다. 80행으로 추정한 기울기의 흔들림을 줄이는 목적이며, 같은 검증에서 RMSE 2.843 -> 2.763 (20회 중 19회에서 낮음).
- **주의**: 격자는 5가지를 비교해 가장 좋은 것을 골랐으므로 이 개선 폭은 낙관적입니다(다른 격자는 -0.05 ~ +0.08). 실제 생성 계수가 이 값이 아니면 소폭 손해입니다.
- **상한**: 입력이 거의 같은 두 행의 정답이 0.1~4.3 차이 나고, 추정 잡음 표준편차가 약 3.1 입니다. 참 계수를 알아도 RMSE 는 약 2.70(78.1점)이라, 85점(RMSE 1.70)은 이 데이터로는 도달할 수 없습니다. 부스팅 트리(3.99~4.86), 가우시안 프로세스(3.17), 추가 항 13종은 모두 Tobit 보다 나빴습니다."""
src = nb.cells[i_code].source
add = '''
class TobitNice(Tobit):
    """Tobit 적합 후 원 단위 기울기를 격자로 반올림하고 절편·sigma 만 다시 추정 (v2 메인)"""
    def __init__(self, grid=(1.0, 0.5, 0.05)): super().__init__(); self.grid = np.array(grid)
    def fit(self, X, y):
        super().fit(X, y); X = np.asarray(X, float); y = np.asarray(y, float)
        self.sl = np.round((self.b[1:] / self.s) / self.grid) * self.grid; lin = X @ self.sl; cen = y <= self.L + 1e-9
        def nll(w):
            mu = w[0] + lin; sg = np.exp(w[1])
            return -(norm.logpdf((y[~cen] - mu[~cen]) / sg).sum() - (~cen).sum() * np.log(sg) + norm.logcdf((self.L - mu[cen]) / sg).sum())
        r = minimize(nll, [np.mean(y - lin), np.log(self.sg)], method="BFGS"); self.b0, self.sg = float(r.x[0]), float(np.exp(r.x[1])); return self
    def predict(self, X):
        mu = self.b0 + np.asarray(X, float) @ self.sl; z = (mu - self.L) / self.sg
        return self.L + (mu - self.L) * norm.cdf(z) + self.sg * norm.pdf(z)
'''
marker = "features = [c for c in test.columns"
assert marker in src and "TobitNice" not in src
src = src.replace(marker, add.lstrip("\n") + "\n" + marker)
src = src.replace('for n, v in (("선형", a), ("선형 + 하한 30 클리핑", b), ("Tobit (메인)", c)):', 'd = cv_rmse(X, y, TobitNice)\nfor n, v in (("선형", a), ("선형 + 하한 30 클리핑", b), ("Tobit (v1)", c), ("Tobit 기울기 반올림 (메인 v2)", d)):')
src = src.replace('print("Tobit 이 선형+클리핑보다 낮은 반복:", int((c < b).sum()), "/", R)', 'print("Tobit 이 선형+클리핑보다 낮은 반복:", int((c < b).sum()), "/", R, "| v2 가 v1 보다 낮은 반복:", int((d < c).sum()), "/", R)')
src = src.replace('model = Tobit().fit(X, y)\nprint("sigma %.2f" % model.sg)', 'model = TobitNice().fit(X, y)\nprint("기울기(수면, 야간근무, 카페인):", model.sl.tolist(), "| 절편 %.2f | sigma %.2f" % (model.b0, model.sg))')
assert "TobitNice().fit" in src and "메인 v2" in src
nb.cells[i_code].source = src
for c in nb.cells:
    if c.cell_type == "code": c.outputs = []; c.execution_count = None
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": d}}).execute()
nbformat.write(nb, f"{d}/work.ipynb"); shutil.copy(f"{d}/submission.csv", f"{d}/versions/v2_tobit_rounded_main.csv"); print("done")
