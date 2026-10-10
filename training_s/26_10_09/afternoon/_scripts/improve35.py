import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize, nnls
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
R = 20
# =========================== P3 ===========================
t3 = pd.read_csv("problem03/data/train.csv"); F3 = ["leadership_score", "facility_score", "meal_score", "rest_time_score"]; X3 = t3[F3].values; y3 = t3.overall_satisfaction.values
class Blend:
    def fit(self, X, y): self.a = LinearRegression().fit(X, y); self.m = LinearRegression().fit(X.mean(1).reshape(-1, 1), y); return self
    def predict(self, X): return (self.a.predict(X) + self.m.predict(X.mean(1).reshape(-1, 1))) / 2
class NoInt:                       # 절편 없음, 계수 자유 (4개)
    def fit(self, X, y): self.w = np.linalg.lstsq(X, y, rcond=None)[0]; return self
    def predict(self, X): return X @ self.w
class Sum1:                        # 계수 합 1, 절편 없음 (자유 3개) = 가중 평균
    def __init__(self, grid=None, intercept=False): self.grid, self.intercept = grid, intercept
    def fit(self, X, y):
        Z = X[:, :3] - X[:, [3]]; t = y - X[:, 3]
        if self.intercept: A = np.c_[Z, np.ones(len(Z))]; s = np.linalg.lstsq(A, t, rcond=None)[0]; w3, self.b = s[:3], s[3]
        else: w3 = np.linalg.lstsq(Z, t, rcond=None)[0]; self.b = 0.0
        self.w = np.r_[w3, 1 - w3.sum()]
        if self.grid:
            self.w = np.round(self.w / self.grid) * self.grid; self.w[np.argmax(self.w)] += 1 - self.w.sum()      # 반올림 후 합이 1 이 되도록 가장 큰 계수에서 보정
            if self.intercept: self.b = float(np.mean(y - X @ self.w))
        return self
    def predict(self, X): return X @ self.w + self.b
def cvr(mk):
    out = []
    for r in range(R):
        oof = np.zeros(80)
        for a, b in KFold(5, shuffle=True, random_state=r).split(X3): oof[b] = mk().fit(X3[a], y3[a]).predict(X3[b])
        out.append(np.sqrt(np.mean((y3 - oof) ** 2)))
    return np.array(out)
print("== P3: 제약을 건 선형 모델 (5-Fold x 20) ==")
base = cvr(Blend); print("  %-44s RMSE %.4f  점수 %.2f" % ("[메인] 선형+평균선형 평균", base.mean(), 100 / (1 + base.mean() / 1.05889019)))
C3 = {"절편 없음, 계수 자유(4개)": NoInt, "계수 합 1, 절편 없음(3개) = 가중 평균": Sum1, "계수 합 1 + 절편(4개)": lambda: Sum1(intercept=True),
      "가중 평균, 계수를 0.05 단위로 반올림": lambda: Sum1(grid=0.05), "가중 평균, 계수를 0.1 단위로 반올림": lambda: Sum1(grid=0.1)}
r3 = {}
for n, mk in C3.items():
    v = cvr(mk); r3[n] = v; d = v - base; print("  %-44s RMSE %.4f  점수 %.2f  메인 대비 %+.4f (%2d/20 반복에서 더 좋음)" % (n, v.mean(), 100 / (1 + v.mean() / 1.05889019), d.mean(), int((d < 0).sum())))
for n, m in (("OLS(절편 포함)", None), ("절편 없음", NoInt()), ("가중 평균(합 1)", Sum1()), ("합 1 + 절편", Sum1(intercept=True)), ("가중 평균 0.05 반올림", Sum1(grid=0.05))):
    if m is None: l = LinearRegression().fit(X3, y3); print("   전체 적합 %-22s 계수 %s 합 %.3f 절편 %.3f | 잔차 sd %.4f" % (n, l.coef_.round(3).tolist(), l.coef_.sum(), l.intercept_, (y3 - l.predict(X3)).std()))
    else: m.fit(X3, y3); print("   전체 적합 %-22s 계수 %s 합 %.3f 절편 %.3f | 잔차 sd %.4f" % (n, np.round(m.w, 3).tolist(), m.w.sum(), getattr(m, "b", 0.0), (y3 - m.predict(X3)).std()))
# =========================== P5 ===========================
t5 = pd.read_csv("problem05/data/train.csv"); te5 = pd.read_csv("problem05/data/test.csv"); F5 = ["pushup", "run_time", "sleep_quality", "stress_level"]; X5 = t5[F5].values.astype(float); fit = (t5.status == "fit").values; rec = t5.recovery_days.values.astype(float); SIGN = np.array([1, -1, 1, -1])
def box_th(X, y): return (X * SIGN)[y].min(0)
def soft_p(X, y, Xb):
    V = X * SIGN; th = box_th(X, y); lo = th.copy()
    for j in range(4):
        others = np.all(np.delete(V, j, 1) >= np.delete(th, j), axis=1); below = V[(~y) & others & (V[:, j] < th[j]), j]; lo[j] = below.max() if len(below) else V[:, j].min() - 1e-9
    return np.clip((Xb * SIGN - lo) / np.where(th > lo, th - lo, 1.0), 0, 1).prod(1)
def bag_p(X, y, Xb, B=300, seed=0, soft=False):
    """fit 행을 재표집해 임계값을 여러 번 추정하고, 통과한 비율을 fit 확률로 사용"""
    rng = np.random.default_rng(seed); fi = np.flatnonzero(y); ui = np.flatnonzero(~y); P = np.zeros(len(Xb))
    for _ in range(B):
        i = np.r_[rng.choice(fi, len(fi)), rng.choice(ui, len(ui))]
        P += soft_p(X[i], y[i], Xb) if soft else ((Xb * SIGN) >= box_th(X[i], y[i])).all(1)
    return P / B
cv5 = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0).split(X5, fit))
def run(pfun, status_thr=0.5, hedge=True):
    acc, rm = [], []
    for r in range(R):
        ps = np.zeros(80, bool); oof = np.zeros(80)
        for a, b in cv5[r * 5:(r + 1) * 5]:
            p = pfun(X5[a], fit[a], X5[b]); mf, mu = rec[a][fit[a]].mean(), rec[a][~fit[a]].mean(); ps[b] = p > status_thr
            oof[b] = p * mf + (1 - p) * mu if hedge else np.where(p > status_thr, mf, mu)
        acc.append(np.mean(ps == fit)); rm.append(np.sqrt(np.mean((rec - oof) ** 2)))
    acc, rm = np.array(acc), np.array(rm); return acc, rm, 50 * acc + 50 / (1 + rm / 2.46475151)
print("\n== P5: 임계값 불확실성 반영 (Stratified 5-Fold x 20) ==")
hard = lambda X, y, Xb: ((Xb * SIGN) >= box_th(X, y)).all(1).astype(float)
b5 = run(hard, hedge=False); print("  %-52s 정확도 %.4f  RMSE %.4f  점수 %.2f" % ("상자 극값 (v1)", b5[0].mean(), b5[1].mean(), b5[2].mean()))
C5 = {"[메인 v2] 구간 확률: status p>0.5, 회복일 가중": (soft_p, 0.5, True), "구간 확률: status p>0.3, 회복일 가중": (soft_p, 0.3, True), "구간 확률: status p>0, 회복일 가중": (soft_p, 1e-9, True),
      "재표집 상자 300회: status p>0.5, 회복일 가중": (bag_p, 0.5, True), "재표집 상자: status p>0.3, 회복일 가중": (bag_p, 0.3, True),
      "재표집+구간 확률: status p>0.5, 회복일 가중": (lambda X, y, Xb: bag_p(X, y, Xb, soft=True), 0.5, True), "재표집+구간 확률: status p>0.3, 회복일 가중": (lambda X, y, Xb: bag_p(X, y, Xb, soft=True), 0.3, True)}
for n, (pf, thr, hd) in C5.items():
    a, rmv, sc = run(pf, thr, hd); d = sc - b5[2]; print("  %-52s 정확도 %.4f  RMSE %.4f  점수 %.2f  v1 대비 %+.2f (%2d/20)" % (n, a.mean(), rmv.mean(), sc.mean(), d.mean(), int((d > 0).sum())), flush=True)
Xt = te5[F5].values.astype(float); pb = bag_p(X5, fit, Xt, B=2000, soft=True); ps = soft_p(X5, fit, Xt); ph = bag_p(X5, fit, Xt, B=2000)
print("  테스트에서 확률이 0 과 1 사이인 행 (구간 / 재표집 / 재표집+구간):"); m = (pb > 0.001) & (pb < 0.999) | (ps > 0) & (ps < 1)
print(te5.assign(p_구간=ps.round(3), p_재표집=ph.round(3), p_재표집구간=pb.round(3))[m | (pb > 0.5)].to_string(index=False))
