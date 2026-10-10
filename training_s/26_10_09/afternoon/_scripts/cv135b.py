import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

R = 20
load = lambda p: pd.read_csv(f"problem{p}/data/train.csv")


class Tobit:
    """하한 L 에서 검열된 선형 회귀 (최대우도). predict: mode='clip' -> max(L, mu), mode='mean' -> E[max(L, mu+eps)]"""
    def __init__(self, L=30.0, mode="mean"): self.L, self.mode = L, mode
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); self.m, self.s = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.m) / self.s]; cen = y <= self.L + 1e-9
        b0 = np.linalg.lstsq(Z, y, rcond=None)[0]; s0 = np.log(max(np.std(y - Z @ b0), 1e-2))
        def nll(w):
            b, sg = w[:-1], np.exp(w[-1]); mu = Z @ b
            return -(norm.logpdf((y[~cen] - mu[~cen]) / sg).sum() - (~cen).sum() * np.log(sg) + norm.logcdf((self.L - mu[cen]) / sg).sum())
        r = minimize(nll, np.r_[b0, s0], method="BFGS"); self.b, self.sg = r.x[:-1], float(np.exp(r.x[-1])); return self
    def predict(self, X):
        Z = np.c_[np.ones(len(X)), (np.asarray(X, float) - self.m) / self.s]; mu = Z @ self.b
        if self.mode == "clip": return np.maximum(self.L, mu)
        z = (mu - self.L) / self.sg; return self.L + (mu - self.L) * norm.cdf(z) + self.sg * norm.pdf(z)


def cv_reg(X, y, mk, post=None, name=""):
    rm = []
    for r in range(R):
        oof = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=r).split(X):
            p = mk().fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b]); oof[b] = post(p) if post else p
        rm.append(np.sqrt(np.mean((y.values - oof) ** 2)))
    rm = np.array(rm); print(f"  {name:46s} RMSE {rm.mean():.4f} (반복 sd {rm.std():.4f})", flush=True); return rm

def paired(a, b, la, lb):
    d = a - b; print(f"    [{la}] - [{lb}] 차 평균 {d.mean():+.4f} (sd {d.std():.4f}, {int((d < 0).sum())}/{len(d)} 반복에서 앞이 더 좋음)")


if __name__ == "__main__":
    # ---------- P1 ----------
    t = load("01"); y = t.fatigue_score; F = ["avg_sleep_hours", "night_shift_days", "caffeine_intake"]
    print("== P1 ==")
    lc = cv_reg(t[F], y, LinearRegression, lambda p: np.clip(p, 30, None), "선형 + 하한 30 클리핑 (현재 최선)")
    tc = cv_reg(t[F], y, lambda: Tobit(mode="clip"), None, "Tobit, 예측 max(30, mu)")
    tm = cv_reg(t[F], y, lambda: Tobit(mode="mean"), None, "Tobit, 예측 E[max(30, mu+eps)]")
    paired(tm, lc, "Tobit 기대값", "선형+클립"); paired(tm, tc, "Tobit 기대값", "Tobit 클립")
    m = Tobit().fit(t[F], y); print("    전체 적합: sigma %.2f | 표준화 계수" % m.sg, dict(zip(["절편"] + F, m.b.round(2))))
    # ---------- P3 ----------
    t = load("03"); y = t.overall_satisfaction; F3 = ["leadership_score", "facility_score", "meal_score", "rest_time_score"]
    print("\n== P3 ==")
    class MeanLin:
        def fit(self, X, y): self.m = LinearRegression().fit(X.mean(axis=1).values.reshape(-1, 1), y); return self
        def predict(self, X): return self.m.predict(X.mean(axis=1).values.reshape(-1, 1))
    class Blend:
        def fit(self, X, y): self.a = LinearRegression().fit(X, y); self.b = MeanLin().fit(X, y); return self
        def predict(self, X): return (self.a.predict(X) + self.b.predict(X)) / 2
    l3 = cv_reg(t[F3], y, LinearRegression, None, "선형 (계수 4개+절편)")
    ml = cv_reg(t[F3], y, MeanLin, None, "평균 한 변수 선형 (기울기+절편)")
    bl = cv_reg(t[F3], y, Blend, None, "두 모델 평균")
    r1 = cv_reg(t[F3], y, LinearRegression, lambda p: np.round(p, 1), "선형 + 소수 1자리 반올림")
    paired(ml, l3, "평균 선형", "선형"); paired(bl, l3, "블렌드", "선형")
    # ---------- P5 ----------
    t = load("05"); F5 = ["pushup", "run_time", "sleep_quality", "stress_level"]; X = t[F5]; fit = (t.status == "fit").values; rec = t.recovery_days.values.astype(float)
    SIGN = np.array([1, -1, 1, -1])            # fit 방향: pushup 높을수록, run_time 낮을수록, sleep 높을수록, stress 낮을수록
    class Box:
        """fit = 네 기준을 모두 통과. margin=0: 학습 fit 의 극값을 임계값으로, margin=0.5: 그 극값과 (다른 세 기준은 통과한) 가장 가까운 unfit 사이의 중간"""
        def __init__(self, margin=0.0): self.margin = margin
        def fit(self, X, y):
            V = np.asarray(X, float) * SIGN; y = np.asarray(y, bool); self.th = V[y].min(0).copy()
            if self.margin > 0:
                for j in range(4):
                    others = np.all(np.delete(V, j, 1) >= np.delete(self.th, j), axis=1)
                    below = V[(~y) & others & (V[:, j] < self.th[j]), j]
                    if len(below): self.th[j] = self.th[j] - self.margin * (self.th[j] - below.max())
            return self
        def nfail(self, X): return (np.asarray(X, float) * SIGN < self.th).sum(1)
        def predict(self, X): return self.nfail(X) == 0
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0).split(X, fit))
    print("\n== P5 ==")
    def run(mk, name, proba=False):
        acc, rm, rm_soft = [], [], []
        for r in range(R):
            ps = np.zeros(len(t), bool); oof = np.zeros(len(t))
            for a, b in cv[r * 5:(r + 1) * 5]:
                m = mk().fit(X.iloc[a], fit[a]); pb = m.predict(X.iloc[b]).astype(bool); ps[b] = pb
                mf, mu = rec[a][fit[a]].mean(), rec[a][~fit[a]].mean()
                if proba: p = m.predict_proba(X.iloc[b])[:, 1]; oof[b] = p * mf + (1 - p) * mu
                else: oof[b] = np.where(pb, mf, mu)
            acc.append(np.mean(ps == fit)); rm.append(np.sqrt(np.mean((rec - oof) ** 2)))
        acc, rm = np.array(acc), np.array(rm); sc = 50 * acc + 50 / (1 + rm / 2.46475151)
        print(f"  {name:46s} 정확도 {acc.mean():.4f} | 회복 RMSE {rm.mean():.4f} | 점수 {sc.mean():.2f} (sd {sc.std():.2f})", flush=True); return sc
    s_lr = run(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)), "로지스틱 C=1, 확률 혼합 (이전 최선)", proba=True)
    run(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)), "로지스틱 C=1, 예측 status 그룹 평균")
    run(lambda: DecisionTreeClassifier(max_depth=4, random_state=0), "결정트리 depth 4, 그룹 평균")
    s_b0 = run(lambda: Box(0.0), "상자 규칙 (fit 극값), 그룹 평균")
    s_b5 = run(lambda: Box(0.5), "상자 규칙 (중간 여유), 그룹 평균")
    paired(-s_b0, -s_lr, "상자 극값", "로지스틱"); paired(-s_b5, -s_b0, "상자 중간", "상자 극값")
    # 기준 미달 개수와 회복일의 관계 (전체 데이터)
    bx = Box(0.0).fit(X, fit); nf = bx.nfail(X)
    print("   임계값(pushup>=, run_time<=, sleep>=, stress<=):", (bx.th * SIGN).round(2).tolist())
    print("   기준 미달 개수별 회복일 평균:", pd.Series(rec).groupby(nf).agg(["mean", "std", "count"]).round(2).to_dict("index"))
