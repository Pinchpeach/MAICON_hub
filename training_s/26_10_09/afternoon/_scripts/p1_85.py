import sys, os, warnings, itertools; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as tdist
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, DotProduct
from cv135b import Tobit, cv_reg, paired, R
t = pd.read_csv("problem01/data/train.csv"); y = t.fatigue_score; F = ["avg_sleep_hours", "night_shift_days", "caffeine_intake"]; X = t[F]; yv = y.values
S = lambda r: 100 / (1 + r / 9.65650938)
print("목표 85점에 필요한 RMSE: %.3f | 80점: %.3f | 현재 2.843 -> %.2f점" % (9.65650938 * (100 / 85 - 1), 9.65650938 * 0.25, S(2.843)))
# ---------- 1. 모델 가정 없는 잡음 하한: 입력이 거의 같은 두 행의 타깃 차이 ----------
print("\n== 1. 모델 가정 없는 잡음 추정 (입력이 가까운 행 쌍의 타깃 차이) ==")
unc = yv > 30; Z = ((X - X.mean()) / X.std()).values[unc]; yu = yv[unc]; D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)); iu = np.triu_indices(len(yu), 1); d = D[iu]; dy2 = (yu[iu[0]] - yu[iu[1]]) ** 2
for q in (0.01, 0.02, 0.05, 0.10):
    m = d <= np.quantile(d, q); print("  가장 가까운 쌍 %4.0f%% (%3d쌍, 표준화 거리 <= %.2f): 추정 잡음 sd %.2f" % (q * 100, m.sum(), np.quantile(d, q), np.sqrt(dy2[m].mean() / 2)))
print("  (30 초과 65행만 사용. 가까운 쌍일수록 '같은 입력인데 다른 정답'에 가까움. 거리가 0이 아니라 약간 과대 추정)")
same = [(i, j) for i, j in zip(*iu) if D[i, j] < 0.35]; print("  가장 가까운 5쌍의 입력과 정답:")
for i, j in sorted(same, key=lambda p: D[p])[:5]: print("    ", t[unc].iloc[i][F].tolist(), yu[i], "|", t[unc].iloc[j][F].tolist(), yu[j], "| 차이 %.1f" % abs(yu[i] - yu[j]))
# ---------- 2. Tobit 을 유지한 변형 ----------
print("\n== 2. Tobit 유지 변형 (5-Fold x 20) ==")
base = cv_reg(X, y, Tobit, None, "Tobit (메인)")
class TobitT(Tobit):
    """잡음을 정규분포 대신 t 분포(두꺼운 꼬리)로 둔 검열 회귀"""
    def __init__(self, df=5): super().__init__(); self.df = df
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); self.m, self.s = X.mean(0), X.std(0) + 1e-9; Zm = np.c_[np.ones(len(X)), (X - self.m) / self.s]; cen = y <= 30 + 1e-9
        b0 = np.linalg.lstsq(Zm, y, rcond=None)[0]; s0 = np.log(max(np.std(y - Zm @ b0), 1e-2))
        def nll(w):
            b, sg = w[:-1], np.exp(w[-1]); mu = Zm @ b
            return -(tdist.logpdf((y[~cen] - mu[~cen]) / sg, self.df).sum() - (~cen).sum() * np.log(sg) + tdist.logcdf((30 - mu[cen]) / sg, self.df).sum())
        r = minimize(nll, np.r_[b0, s0], method="BFGS"); self.b, self.sg = r.x[:-1], float(np.exp(r.x[-1])); return self
    def predict(self, X):
        Zm = np.c_[np.ones(len(X)), (np.asarray(X, float) - self.m) / self.s]; mu = Zm @ self.b; g = np.linspace(-8, 8, 801); w = tdist.pdf(g, self.df); w /= w.sum()
        return (np.maximum(30, mu[:, None] + self.sg * g[None, :]) * w).sum(1)
for df in (3, 5, 10):
    r = cv_reg(X, y, lambda df=df: TobitT(df), None, f"Tobit t분포 잡음 자유도 {df}"); paired(r, base, f"t{df}", "메인")
class TobitNice(Tobit):
    def __init__(self, grid=(1.0, 0.5, 0.05)): super().__init__(); self.grid = np.array(grid)
    def fit(self, X, y):
        super().fit(X, y); X = np.asarray(X, float); y = np.asarray(y, float); sl = np.round((self.b[1:] / self.s) / self.grid) * self.grid; lin = X @ sl; cen = y <= 30 + 1e-9
        def nll(w):
            mu = w[0] + lin; sg = np.exp(w[1]); return -(norm.logpdf((y[~cen] - mu[~cen]) / sg).sum() - (~cen).sum() * np.log(sg) + norm.logcdf((30 - mu[cen]) / sg).sum())
        r = minimize(nll, [np.mean(y - lin), np.log(self.sg)], method="BFGS"); self.sl, self.b0, self.sg = sl, r.x[0], float(np.exp(r.x[1])); return self
    def predict(self, X):
        mu = self.b0 + np.asarray(X, float) @ self.sl; z = (mu - 30) / self.sg; return 30 + (mu - 30) * norm.cdf(z) + self.sg * norm.pdf(z)
nice = cv_reg(X, y, TobitNice, None, "Tobit 기울기 반올림 (1, 0.5, 0.05)"); paired(nice, base, "반올림", "메인")
# 변환 라이브러리에서 추가 항 1~2개를 고르는 탐색 (전체 데이터로 고르므로 낙관적: 이것조차 못 내려가면 하한이라는 근거)
lib = {"1/수면": lambda d: 1 / d.avg_sleep_hours, "log수면": lambda d: np.log(d.avg_sleep_hours), "(8-수면)+^2": lambda d: np.maximum(0, 8 - d.avg_sleep_hours) ** 2, "sqrt야간": lambda d: np.sqrt(d.night_shift_days), "야간>=3": lambda d: (d.night_shift_days >= 3) * 1.0,
       "log카페인": lambda d: np.log(d.caffeine_intake), "카페인>200": lambda d: (d.caffeine_intake > 200) * 1.0, "카페인>150": lambda d: (d.caffeine_intake > 150) * 1.0, "수면<6": lambda d: (d.avg_sleep_hours < 6) * 1.0, "수면x야간": lambda d: d.avg_sleep_hours * d.night_shift_days,
       "카페인/수면": lambda d: d.caffeine_intake / d.avg_sleep_hours, "야간/수면": lambda d: d.night_shift_days / d.avg_sleep_hours, "야간x카페인": lambda d: d.night_shift_days * d.caffeine_intake}
out = []
for k, f in lib.items():
    Xa = X.assign(z=f(t)); r = []
    for rep in range(5):
        oof = np.zeros(80)
        for a, b in KFold(5, shuffle=True, random_state=rep).split(Xa): oof[b] = Tobit().fit(Xa.iloc[a], y.iloc[a]).predict(Xa.iloc[b])
        r.append(np.sqrt(np.mean((yv - oof) ** 2)))
    out.append((np.mean(r), k))
b5 = base[:5].mean(); print("  추가 항 1개 탐색(13개, 5회 반복 기준, 메인 %.4f):" % b5, [(k, round(v, 3)) for v, k in sorted(out)[:5]], "... 최악", [(k, round(v, 3)) for v, k in sorted(out)[-1:]])
# ---------- 3. 완전히 유연한 모델: 가우시안 프로세스 ----------
print("\n== 3. 유연한 비모수 모델 (참고) ==")
gp = lambda: GaussianProcessRegressor(C(10.0) * RBF([1.0, 1.0, 1.0]) + C(10.0) * DotProduct() + WhiteKernel(5.0), normalize_y=True, random_state=0)
class Std:
    def __init__(self, mk): self.mk = mk
    def fit(self, X, y): X = np.asarray(X, float); self.m, self.s = X.mean(0), X.std(0); self.g = self.mk().fit((X - self.m) / self.s, y); return self
    def predict(self, X): return self.g.predict((np.asarray(X, float) - self.m) / self.s)
RR = 5; rm = []
for rep in range(RR):
    oof = np.zeros(80)
    for a, b in KFold(5, shuffle=True, random_state=rep).split(X): oof[b] = np.clip(Std(gp).fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b]), 30, None)
    rm.append(np.sqrt(np.mean((yv - oof) ** 2)))
print("  가우시안 프로세스(RBF+선형 커널)+클립, 5회 반복: RMSE %.4f (메인 같은 5회 %.4f)" % (np.mean(rm), b5))
# ---------- 4. 요약 ----------
print("\n== 4. 요약 ==")
for n, v in (("메인 Tobit", base.mean()), ("기울기 반올림", nice.mean()), ("참 모수를 알 때(모의)", 2.702), ("목표 85점", 1.704)): print("  %-18s RMSE %.3f -> %.2f점" % (n, v, S(v)))
