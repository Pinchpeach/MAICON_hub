import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
t = pd.read_csv("problem05/data/train.csv"); te = pd.read_csv("problem05/data/test.csv")
F = ["pushup", "run_time", "sleep_quality", "stress_level"]; SIGN = np.array([1, -1, 1, -1]); fit = (t.status == "fit").values; rec = t.recovery_days.values.astype(float); X = t[F]; R = 20
class SoftBox:
    """기준별 통과 확률: fit 극값 이상이면 1, (다른 세 기준은 통과한) 가장 가까운 unfit 값 이하면 0, 그 사이는 선형. p = 네 확률의 곱"""
    def fit(self, X, y):
        V = np.asarray(X, float) * SIGN; y = np.asarray(y, bool); self.hi = V[y].min(0); self.lo = self.hi.copy()
        for j in range(4):
            others = np.all(np.delete(V, j, 1) >= np.delete(self.hi, j), axis=1); below = V[(~y) & others & (V[:, j] < self.hi[j]), j]
            self.lo[j] = below.max() if len(below) else V[:, j].min() - 1e-9      # 이 기준만 미달한 unfit 이 없으면 구간을 데이터 최솟값까지로 본다
        return self
    def proba(self, X):
        V = np.asarray(X, float) * SIGN; w = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        return np.clip((V - self.lo) / w, 0, 1).prod(1)
cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0).split(X, fit))
def run(mode, name):
    acc, rm = [], []
    for r in range(R):
        ps = np.zeros(len(t), bool); oof = np.zeros(len(t))
        for a, b in cv[r * 5:(r + 1) * 5]:
            m = SoftBox().fit(X.iloc[a], fit[a]); p = m.proba(X.iloc[b]); mf, mu = rec[a][fit[a]].mean(), rec[a][~fit[a]].mean()
            hard = p >= 1.0 - 1e-12
            ps[b] = hard if mode in ("hard", "hard+softrec") else (p > 0.5)
            oof[b] = np.where(hard, mf, mu) if mode == "hard" else (p * mf + (1 - p) * mu if mode in ("hard+softrec", "soft") else np.where(p > 0.5, mf, mu))
        acc.append(np.mean(ps == fit)); rm.append(np.sqrt(np.mean((rec - oof) ** 2)))
    acc, rm = np.array(acc), np.array(rm); sc = 50 * acc + 50 / (1 + rm / 2.46475151)
    print(f"  {name:52s} 정확도 {acc.mean():.4f} | RMSE {rm.mean():.4f} | 점수 {sc.mean():.2f} (sd {sc.std():.2f})"); return sc
a = run("hard", "상자 극값 (현재 메인)")
b = run("hard+softrec", "status 는 극값, 회복일만 확률 가중")
c = run("soft", "status 는 p>0.5, 회복일 확률 가중")
d = run("soft_hardrec", "status 는 p>0.5, 회복일은 그룹 평균")
for n, v in (("회복일만 확률 가중", b), ("둘 다 확률", c), ("status p>0.5", d)):
    df = v - a; print(f"    [{n}] - [메인] 점수 차 {df.mean():+.3f} (sd {df.std():.3f}, {int((df > 0).sum())}/{R} 반복에서 높음)")
m = SoftBox().fit(X, fit); pt = m.proba(te[F])
print("전체 학습 구간: hi", (m.hi * SIGN).tolist(), "lo", (m.lo * SIGN).round(1).tolist())
print("테스트 p_fit > 0 인 행:"); print(te.assign(p_fit=pt.round(3))[pt > 0].to_string(index=False))
