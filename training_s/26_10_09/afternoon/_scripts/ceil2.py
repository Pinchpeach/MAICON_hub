import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, HuberRegressor
from sklearn.model_selection import KFold
from cv135b import Tobit, cv_reg, paired, R
L = lambda p, f="train": pd.read_csv(f"problem{p}/data/{f}.csv")
t1, t3, t5 = L("01"), L("03"), L("05")
F1 = ["avg_sleep_hours", "night_shift_days", "caffeine_intake"]; F3 = ["leadership_score", "facility_score", "meal_score", "rest_time_score"]; F5 = ["pushup", "run_time", "sleep_quality", "stress_level"]
# ---------- 문제 간 연결: 잔차와 다른 문제 변수의 상관 ----------
print("== 문제 간 연결: 각 문제 잔차 vs 다른 문제의 변수 (같은 soldier_id) ==")
allp = {p: pd.concat([L(p), L(p, "test")]).set_index("soldier_id") for p in ("01", "03", "05")}
m1 = Tobit().fit(t1[F1], t1.fatigue_score); Z = np.c_[np.ones(80), (t1[F1].values - m1.m) / m1.s]; mu1 = Z @ m1.b
res = {"01": pd.Series(np.where(t1.fatigue_score > 30, t1.fatigue_score - mu1, np.nan), index=t1.soldier_id),
       "03": pd.Series(t3.overall_satisfaction.values - LinearRegression().fit(t3[F3], t3.overall_satisfaction).predict(t3[F3]), index=t3.soldier_id)}
fit5 = (t5.status == "fit"); res["05"] = pd.Series(np.where(fit5, np.nan, t5.recovery_days - t5.recovery_days[~fit5].mean()), index=t5.soldier_id)
for p, r in res.items():
    out = []
    for q, d in allp.items():
        if q == p: continue
        for c in d.select_dtypes("number").columns:
            j = pd.concat([r, d[c]], axis=1).dropna()
            if len(j) > 20: out.append((f"P{q}.{c}", round(float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]), 3), len(j)))
    out.sort(key=lambda x: -abs(x[1])); print(f"  P{p} 잔차와 상관 상위 3 (검정 {len(out)}개, n≈{out[0][2]}, |r|>0.3 이어야 유의 수준):", out[:3])
# ---------- 값의 소수 자리 분포 ----------
print("\n== P5 값 소수 자리(.0 비율) ==", {c: round(float(((pd.concat([t5, L('05','test')])[c] * 10) % 10 == 0).mean()), 3) for c in ["run_time", "sleep_quality", "stress_level"]})
# ---------- P1: 추정 분산 줄이기 ----------
print("\n== P1 추정 분산 줄이기 (같은 5-Fold x 20) ==")
y1 = t1.fatigue_score
class TobitNice(Tobit):
    """Tobit 적합 후 원 단위 기울기를 정해진 격자로 반올림하고, 절편과 sigma 만 다시 추정"""
    def __init__(self, grid=(0.5, 0.5, 0.01)): super().__init__(); self.grid = np.array(grid)
    def fit(self, X, y):
        super().fit(X, y); X = np.asarray(X, float); y = np.asarray(y, float); sl = self.b[1:] / self.s; sl = np.round(sl / self.grid) * self.grid
        lin = X @ sl; cen = y <= 30 + 1e-9
        def nll(w):
            mu = w[0] + lin; sg = np.exp(w[1]); return -(norm.logpdf((y[~cen] - mu[~cen]) / sg).sum() - (~cen).sum() * np.log(sg) + norm.logcdf((30 - mu[cen]) / sg).sum())
        r = minimize(nll, [np.mean(y - lin), np.log(self.sg)], method="BFGS"); self.sl, self.b0, self.sg = sl, r.x[0], float(np.exp(r.x[1])); return self
    def predict(self, X):
        mu = self.b0 + np.asarray(X, float) @ self.sl; z = (mu - 30) / self.sg; return 30 + (mu - 30) * norm.cdf(z) + self.sg * norm.pdf(z)
class TobitRidge(Tobit):
    def __init__(self, lam=1.0): super().__init__(); self.lam = lam
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); self.m, self.s = X.mean(0), X.std(0) + 1e-9; Zm = np.c_[np.ones(len(X)), (X - self.m) / self.s]; cen = y <= 30 + 1e-9
        b0 = np.linalg.lstsq(Zm, y, rcond=None)[0]; s0 = np.log(max(np.std(y - Zm @ b0), 1e-2))
        def nll(w):
            b, sg = w[:-1], np.exp(w[-1]); mu = Zm @ b
            return -(norm.logpdf((y[~cen] - mu[~cen]) / sg).sum() - (~cen).sum() * np.log(sg) + norm.logcdf((30 - mu[cen]) / sg).sum()) + self.lam * (b[1:] ** 2).sum() / (2 * sg ** 2)
        r = minimize(nll, np.r_[b0, s0], method="BFGS"); self.b, self.sg = r.x[:-1], float(np.exp(r.x[-1])); return self
class TobitBag:
    def __init__(self, n=30): self.n = n
    def fit(self, X, y):
        rng = np.random.default_rng(0); X = np.asarray(X, float); y = np.asarray(y, float); self.ms = []
        for _ in range(self.n):
            i = rng.integers(0, len(X), len(X)); self.ms.append(Tobit().fit(X[i], y[i]))
        return self
    def predict(self, X): return np.mean([m.predict(X) for m in self.ms], axis=0)
base = cv_reg(t1[F1], y1, Tobit, None, "Tobit (메인)")
for g in ((0.5, 0.5, 0.01), (1.0, 0.5, 0.05), (0.5, 0.5, 0.05), (1.0, 1.0, 0.1), (0.25, 0.25, 0.005)):
    r = cv_reg(t1[F1], y1, lambda g=g: TobitNice(g), None, f"Tobit 기울기 반올림 격자 {g}"); paired(r, base, "반올림", "메인")
for lam in (0.3, 1.0, 3.0):
    r = cv_reg(t1[F1], y1, lambda lam=lam: TobitRidge(lam), None, f"Tobit + 릿지 벌점 {lam}"); paired(r, base, "릿지", "메인")
r = cv_reg(t1[F1], y1, TobitBag, None, "Tobit 부트스트랩 평균(30)"); paired(r, base, "배깅", "메인")
full = Tobit().fit(t1[F1], y1); print("   전체 적합 원 단위 기울기:", (full.b[1:] / full.s).round(4).tolist(), "sigma %.3f" % full.sg)
# ---------- P3 ----------
print("\n== P3 추정 분산 줄이기 ==")
y3 = t3.overall_satisfaction
class LinNice:
    def __init__(self, g=0.05): self.g = g
    def fit(self, X, y): m = LinearRegression().fit(X, y); self.c = np.round(m.coef_ / self.g) * self.g; self.b = float(np.mean(y - np.asarray(X) @ self.c)); return self
    def predict(self, X): return np.asarray(X) @ self.c + self.b
class Blend:
    def __init__(self, w=0.5): self.w = w
    def fit(self, X, y): self.a = LinearRegression().fit(X, y); self.m = LinearRegression().fit(X.mean(axis=1).values.reshape(-1, 1), y); return self
    def predict(self, X): return self.w * self.a.predict(X) + (1 - self.w) * self.m.predict(X.mean(axis=1).values.reshape(-1, 1))
b3 = cv_reg(t3[F3], y3, Blend, None, "선형+평균선형 평균 (메인)")
for g in (0.05, 0.1, 0.25):
    r = cv_reg(t3[F3], y3, lambda g=g: LinNice(g), None, f"선형 계수 반올림 격자 {g}"); paired(r, b3, "반올림", "메인")
for w in (0.3, 0.7):
    r = cv_reg(t3[F3], y3, lambda w=w: Blend(w), None, f"블렌드 가중 선형 {w}"); paired(r, b3, f"w={w}", "메인")
r = cv_reg(t3[F3], y3, BayesianRidge, None, "BayesianRidge"); paired(r, b3, "BayesianRidge", "메인")
r = cv_reg(t3[F3], y3, lambda: HuberRegressor(epsilon=1.5), None, "Huber"); paired(r, b3, "Huber", "메인")
