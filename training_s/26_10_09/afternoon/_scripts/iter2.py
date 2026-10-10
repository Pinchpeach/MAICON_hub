import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from cv135b import Tobit, cv_reg, paired, R
from sklearn.model_selection import RepeatedStratifiedKFold
# ================= P5: 임계값 허용 구간 =================
t = pd.read_csv("problem05/data/train.csv"); te = pd.read_csv("problem05/data/test.csv")
F = ["pushup", "run_time", "sleep_quality", "stress_level"]; SIGN = np.array([1, -1, 1, -1]); fit = (t.status == "fit").values
V = t[F].values * SIGN; tight = V[fit].min(0); loose = tight.copy()
print("== P5 임계값 허용 구간 (다른 세 기준은 통과한 unfit 중 가장 가까운 값 ~ fit 극값) ==")
for j, c in enumerate(F):
    others = np.all(np.delete(V, j, 1) >= np.delete(tight, j), axis=1); below = V[(~fit) & others & (V[:, j] < tight[j]), j]
    lo = below.max() if len(below) else -np.inf; loose[j] = lo
    a, b = sorted([lo * SIGN[j], tight[j] * SIGN[j]])
    print(f"  {c:14s} fit 극값 {tight[j]*SIGN[j]:6.1f} | 이 기준만 미달한 unfit {len(below)}명, 가장 가까운 값 {lo*SIGN[j] if len(below) else float('nan'):6.1f} | 실제 임계값은 {a:.1f} ~ {b:.1f} 사이")
    print("     이 기준만 미달한 unfit 값들:", np.sort(below * SIGN[j]).round(1).tolist())
# 학습 데이터에서 fit 근처 값 분포
for j, c in enumerate(F): print(f"  {c} fit 값:", np.sort(t[fit][c].values).tolist())
# 테스트에서 애매한 구간에 있는 행
Vt = te[F].values * SIGN; p_tight = (Vt >= tight).all(1)
print("\n테스트 21행: 상자(극값) 통과", int(p_tight.sum()), "명")
near = []
for i in range(len(te)):
    fails = [(F[j], round(float(te[F].values[i, j]), 1), round(float(tight[j] * SIGN[j]), 1)) for j in range(4) if Vt[i, j] < tight[j]]
    if len(fails) == 1: near.append((te.soldier_id[i], fails[0]))
print("기준 하나만 미달한 테스트 행 (특징, 값, 임계값):"); [print("   ", n) for n in near]
print("테스트 기준 미달 개수 분포:", pd.Series((Vt < tight).sum(1)).value_counts().sort_index().to_dict(), "| 학습:", pd.Series((V < tight).sum(1)).value_counts().sort_index().to_dict())
# 둥근 임계값 후보를 CV 로 비교
class BoxFixed:
    def __init__(self, th): self.th = np.array(th, float) * SIGN
    def fit(self, X, y): return self
    def predict(self, X): return ((np.asarray(X, float) * SIGN) >= self.th).all(1)
rec = t.recovery_days.values.astype(float)
def score_fixed(th):
    p = BoxFixed(th).predict(t[F]); acc = np.mean(p == fit)
    return acc, int((p & ~fit).sum()), int((~p & fit).sum())
print("\n고정 임계값 후보의 학습 80행 정확도 (거짓 fit, 놓친 fit):")
for th in ([38, 9.6, 5.2, 5.8], [35, 10.0, 5.0, 6.0], [40, 9.5, 5.0, 6.0], [35, 9.6, 5.0, 6.0], [38, 10.0, 5.0, 6.0], [36, 9.8, 5.0, 6.0]):
    print("  ", th, score_fixed(th))
# ================= P1: 교호작용 =================
t1 = pd.read_csv("problem01/data/train.csv"); y = t1.fatigue_score; F1 = ["avg_sleep_hours", "night_shift_days", "caffeine_intake"]
print("\n== P1 Tobit 교호작용·비선형 ==")
base = cv_reg(t1[F1], y, Tobit, None, "Tobit 기본 (메인)")
def add(df, cols):
    X = df[F1].copy()
    for n, f in cols.items(): X[n] = f(df)
    return X
cands = {
 "+ 수면x야간": {"sxn": lambda d: d.avg_sleep_hours * d.night_shift_days},
 "+ 수면x카페인": {"sxc": lambda d: d.avg_sleep_hours * d.caffeine_intake},
 "+ 야간x카페인": {"nxc": lambda d: d.night_shift_days * d.caffeine_intake},
 "+ 수면^2": {"s2": lambda d: d.avg_sleep_hours ** 2},
 "+ 야간^2": {"n2": lambda d: d.night_shift_days ** 2},
 "+ 카페인^2": {"c2": lambda d: d.caffeine_intake ** 2},
 "+ 수면부족 max(0,7-수면)": {"def": lambda d: np.maximum(0, 7 - d.avg_sleep_hours)},
 "+ 수면부족 max(0,6-수면)": {"def": lambda d: np.maximum(0, 6 - d.avg_sleep_hours)},
}
for n, c in cands.items():
    r = cv_reg(add(t1, c), y, Tobit, None, "Tobit " + n); paired(r, base, n, "기본")
# 잔차 구조 확인 (전체 적합, 비검열 행)
m = Tobit().fit(t1[F1], y); Z = np.c_[np.ones(len(t1)), (t1[F1].values - m.m) / m.s]; mu = Z @ m.b; unc = y.values > 30; res = y.values[unc] - mu[unc]
print("비검열 행 잔차: sd %.2f | 잔차와 mu 상관 %.3f | |잔차|와 mu 상관(이분산) %.3f | 최대 |잔차| %.1f" % (res.std(), np.corrcoef(res, mu[unc])[0, 1], np.corrcoef(np.abs(res), mu[unc])[0, 1], np.abs(res).max()))
print("원 단위 계수: 수면 %.3f/시간, 야간 %.3f/일, 카페인 %.4f/mg, 절편 %.2f" % (m.b[1]/m.s[0], m.b[2]/m.s[1], m.b[3]/m.s[2], m.b[0] - (m.b[1:]*m.m/m.s).sum()))
