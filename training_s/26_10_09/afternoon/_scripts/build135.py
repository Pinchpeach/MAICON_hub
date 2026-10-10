import nbformat, os, shutil
from nbclient import NotebookClient

CV_HELP = '''from sklearn.model_selection import KFold
R = 20   # 5-Fold 를 20번 반복 (학습 80행뿐이라 단일 분할 점수는 변동이 큼)
def cv_rmse(X, y, make, post=None):
    out = []
    for r in range(R):
        oof = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=r).split(X):
            p = make().fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b]); oof[b] = post(p) if post else p
        out.append(np.sqrt(np.mean((y.values - oof) ** 2)))
    return np.array(out)
'''

P = {}
P["01"] = ("""## 풀이: 하한 30 에서 검열된 선형 회귀 (Tobit)
- **데이터 구조**: 학습 80행 중 15행(19%)의 `fatigue_score`가 정확히 30.0 입니다. 30 초과 행만으로 선형 적합을 하면 이 15행의 예측값이 대부분 30 미만(16~33)으로 나와, 타깃이 `max(30, 선형 + 잡음)` 형태로 만들어졌다고 판단했습니다.
- **모델**: 30 이하를 "검열된 관측"으로 다루는 최대우도 선형 회귀(Tobit)를 직접 구현했습니다(scipy). 예측은 `E[max(30, mu + eps)]` 입니다.
- **검증**(5-Fold x 20회 반복 RMSE): 제공 베이스라인 RF 5.12, 선형 3.74, 선형+하한 클리핑 3.26, **Tobit 2.84**. Tobit 은 20회 반복 모두에서 선형+클리핑보다 낮았습니다.
- 기각: 카페인·야간근무 포화 변환(3.5~4.5), 2차 다항 Ridge(3.39), HGB(5.87).""",
CV_HELP + '''from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

class Tobit:
    """하한 L 에서 검열된 선형 회귀(최대우도). 예측 = E[max(L, mu + eps)]"""
    def __init__(self, L=30.0): self.L = L
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); self.m, self.s = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.m) / self.s]; cen = y <= self.L + 1e-9
        b0 = np.linalg.lstsq(Z, y, rcond=None)[0]; s0 = np.log(max(np.std(y - Z @ b0), 1e-2))
        def nll(w):
            b, sg = w[:-1], np.exp(w[-1]); mu = Z @ b
            return -(norm.logpdf((y[~cen] - mu[~cen]) / sg).sum() - (~cen).sum() * np.log(sg) + norm.logcdf((self.L - mu[cen]) / sg).sum())
        r = minimize(nll, np.r_[b0, s0], method="BFGS"); self.b, self.sg = r.x[:-1], float(np.exp(r.x[-1])); return self
    def predict(self, X):
        Z = np.c_[np.ones(len(X)), (np.asarray(X, float) - self.m) / self.s]; mu = Z @ self.b; z = (mu - self.L) / self.sg
        return self.L + (mu - self.L) * norm.cdf(z) + self.sg * norm.pdf(z)

features = [c for c in test.columns if c != "soldier_id"]; X, y = train[features], train["fatigue_score"]
print("fatigue_score == 30.0 인 행:", int((y == 30).sum()), "/", len(y))
a = cv_rmse(X, y, LinearRegression); b = cv_rmse(X, y, LinearRegression, lambda p: np.clip(p, 30, None)); c = cv_rmse(X, y, Tobit)
for n, v in (("선형", a), ("선형 + 하한 30 클리핑", b), ("Tobit (메인)", c)):
    print(f"{n:22s} RMSE {v.mean():.4f} (반복 sd {v.std():.4f}) -> 점수 {100 / (1 + v.mean() / 9.65650938):.2f}")
print("Tobit 이 선형+클리핑보다 낮은 반복:", int((c < b).sum()), "/", R)

model = Tobit().fit(X, y)
print("sigma %.2f" % model.sg)
submission = pd.DataFrame({"soldier_id": test.soldier_id, "fatigue_score": model.predict(test[features])})[sample.columns]
assert np.isfinite(submission.fatigue_score).all()
submission.to_csv("submission.csv", index=False)
print("예측 범위 %.1f ~ %.1f (train %.1f ~ %.1f)" % (submission.fatigue_score.min(), submission.fatigue_score.max(), y.min(), y.max())); submission.head()''',
"v1_tobit_main.csv")

P["03"] = ("""## 풀이: 선형 회귀와 '4항목 평균' 선형 회귀의 평균
- **데이터 구조**: 네 항목 점수의 선형 결합에 잡음이 더해진 형태입니다(훈련 R2 0.83, 잔차 표준편차 0.44). 계수가 0.33/0.27/0.25/0.24 로 비슷하고 입력끼리 상관이 거의 없습니다.
- **모델**: ① 계수 4개를 자유롭게 두는 선형 회귀와 ② 4항목 평균 하나만 쓰는 선형 회귀(같은 가중치로 수축)의 예측을 평균합니다. 80행으로 계수 4개를 추정하는 불안정을 줄이는 목적입니다.
- **검증**(5-Fold x 20회 반복 RMSE): 제공 베이스라인 RF 0.696, 선형 0.4628, 평균 선형 0.4647, **두 모델 평균 0.4585**. 선형 대비 차이는 0.004 로 작고(20회 중 17회에서 낮음), 잔차 표준편차 0.44 가 사실상 하한이라 더 줄이기 어렵습니다.
- 기각: Ridge(0.465~0.484), 2차 다항(0.466), HGB(0.649), 반올림 후처리(0.464).""",
CV_HELP + '''from sklearn.linear_model import LinearRegression

class MeanLin:
    """4항목 평균 한 변수로 선형 회귀"""
    def fit(self, X, y): self.m = LinearRegression().fit(X.mean(axis=1).values.reshape(-1, 1), y); return self
    def predict(self, X): return self.m.predict(X.mean(axis=1).values.reshape(-1, 1))
class Blend:
    def fit(self, X, y): self.a = LinearRegression().fit(X, y); self.b = MeanLin().fit(X, y); return self
    def predict(self, X): return (self.a.predict(X) + self.b.predict(X)) / 2

features = [c for c in test.columns if c != "soldier_id"]; X, y = train[features], train["overall_satisfaction"]
a = cv_rmse(X, y, LinearRegression); b = cv_rmse(X, y, MeanLin); c = cv_rmse(X, y, Blend)
for n, v in (("선형", a), ("평균 한 변수 선형", b), ("두 모델 평균 (메인)", c)):
    print(f"{n:22s} RMSE {v.mean():.4f} (반복 sd {v.std():.4f}) -> 점수 {100 / (1 + v.mean() / 1.05889019):.2f}")
print("메인이 선형보다 낮은 반복:", int((c < a).sum()), "/", R)

model = Blend().fit(X, y)
print("선형 계수:", dict(zip(features, model.a.coef_.round(3))), "절편 %.3f" % model.a.intercept_)
submission = pd.DataFrame({"soldier_id": test.soldier_id, "overall_satisfaction": model.predict(test[features])})[sample.columns]
assert np.isfinite(submission.overall_satisfaction).all()
submission.to_csv("submission.csv", index=False)
print("예측 범위 %.2f ~ %.2f (train %.1f ~ %.1f)" % (submission.overall_satisfaction.min(), submission.overall_satisfaction.max(), y.min(), y.max())); submission.head()''',
"v1_blend_linear_main.csv")

P["05"] = ("""## 풀이: 네 기준을 모두 통과하면 fit (상자 규칙) + status 별 회복일 평균
- **데이터 구조**: fit 은 10/80(12.5%)뿐이고, fit 10명은 모두 `pushup >= 38`, `run_time <= 9.6`, `sleep_quality >= 5.2`, `stress_level <= 5.8` 안에 있으며 이 상자 안에 unfit 은 0명입니다. 선형 경계(로지스틱)보다 "네 기준 모두 통과" 규칙이 맞는 형태입니다.
- **회복일**: status 가 거의 결정합니다(fit 평균 5.5일, unfit 평균 10.1일). unfit 안에서는 네 특징과의 상관이 -0.20~0.09 로 약하고, 기준 미달 개수별 평균도 9.5~10.6 으로 차이가 작아 **status 별 평균**만 씁니다.
- **검증**(Stratified 5-Fold x 20회 반복): 제공 베이스라인 RF 정확도 0.937, 로지스틱 0.9475 / 회복 RMSE 2.24 / 점수 73.6, **상자 규칙 0.967 / 2.17 / 74.9**(20회 중 18회에서 로지스틱보다 높음). 정답 status 를 알 때의 회복 RMSE 는 1.97 이 하한입니다.
- **주의**: 임계값은 학습 fit 10명의 극값이라 표본이 적습니다. 테스트에 그 극값보다 조금 바깥의 fit 이 있으면 unfit 으로 잘못 분류합니다. 극값과 가장 가까운 unfit 사이의 중간으로 여유를 주는 변형은 점수가 74.5 로 차이가 노이즈 수준이라 쓰지 않았습니다.""",
'''from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

features = [c for c in test.columns if c != "soldier_id"]          # pushup, run_time, sleep_quality, stress_level
SIGN = np.array([1, -1, 1, -1])                                    # fit 방향: pushup↑, run_time↓, sleep_quality↑, stress_level↓
assert features == ["pushup", "run_time", "sleep_quality", "stress_level"]

class Box:
    """네 기준을 모두 통과하면 fit. 임계값 = 학습 데이터 fit 의 극값"""
    def fit(self, X, y): self.th = (np.asarray(X, float) * SIGN)[np.asarray(y, bool)].min(0); return self
    def predict(self, X): return ((np.asarray(X, float) * SIGN) >= self.th).all(1)

X = train[features]; is_fit = (train.status == "fit").values; rec = train.recovery_days.values.astype(float)
R = 20; cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0).split(X, is_fit))
def evaluate(make):
    acc, rm = [], []
    for r in range(R):
        ps = np.zeros(len(train), bool); oof = np.zeros(len(train))
        for a, b in cv[r * 5:(r + 1) * 5]:
            pb = np.asarray(make().fit(X.iloc[a], is_fit[a]).predict(X.iloc[b])).astype(bool); ps[b] = pb
            oof[b] = np.where(pb, rec[a][is_fit[a]].mean(), rec[a][~is_fit[a]].mean())
        acc.append(np.mean(ps == is_fit)); rm.append(np.sqrt(np.mean((rec - oof) ** 2)))
    acc, rm = np.array(acc), np.array(rm); return acc, rm, 50 * acc + 50 / (1 + rm / 2.46475151)
print("fit 비율: %.3f (다수 클래스만 찍으면 정확도 %.3f)" % (is_fit.mean(), 1 - is_fit.mean()))
res = {"로지스틱 C=1": evaluate(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))), "상자 규칙 (메인)": evaluate(Box)}
for n, (acc, rm, sc) in res.items(): print(f"{n:16s} 정확도 {acc.mean():.4f} | 회복 RMSE {rm.mean():.4f} | 점수 {sc.mean():.2f} (반복 sd {sc.std():.2f})")
print("상자 규칙이 로지스틱보다 높은 반복:", int((res["상자 규칙 (메인)"][2] > res["로지스틱 C=1"][2]).sum()), "/", R)

box = Box().fit(X, is_fit)
print("임계값: pushup >= %.0f, run_time <= %.1f, sleep_quality >= %.1f, stress_level <= %.1f" % tuple(box.th * SIGN))
pred_fit = box.predict(test[features]); mean_fit, mean_unfit = rec[is_fit].mean(), rec[~is_fit].mean()
submission = pd.DataFrame({"soldier_id": test.soldier_id, "status": np.where(pred_fit, "fit", "unfit"), "recovery_days": np.where(pred_fit, mean_fit, mean_unfit)})[sample.columns]
assert set(submission.status) <= {"fit", "unfit"} and (submission.recovery_days >= 0).all() and np.isfinite(submission.recovery_days).all()
submission.to_csv("submission.csv", index=False)
print("회복일 평균: fit %.2f, unfit %.2f | 테스트 fit 예측 %d/%d (%.3f, train %.3f)" % (mean_fit, mean_unfit, pred_fit.sum(), len(test), pred_fit.mean(), is_fit.mean())); submission.head()''',
"v1_box_rule_groupmean_main.csv")

root = os.getcwd()
for p, (md, code, vname) in P.items():
    d = os.path.join(root, f"problem{p}"); os.makedirs(os.path.join(d, "versions"), exist_ok=True)
    orig = os.path.join(d, "versions", "notebook_orig.ipynb")
    if not os.path.exists(orig): shutil.copy(os.path.join(d, "work.ipynb"), orig)
    nb = nbformat.read(orig, as_version=4)
    nb.cells = nb.cells[:2] + [nbformat.v4.new_markdown_cell(md), nbformat.v4.new_code_cell(code)] + nb.cells[3:]
    for c in nb.cells:
        if c.cell_type == "code": c.outputs = []; c.execution_count = None
    NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": d}}).execute()
    nbformat.write(nb, os.path.join(d, "work.ipynb"))
    shutil.copy(os.path.join(d, "submission.csv"), os.path.join(d, "versions", vname))
    print("done", p, flush=True)
