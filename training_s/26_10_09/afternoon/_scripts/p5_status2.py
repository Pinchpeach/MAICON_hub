import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut

t = pd.read_csv("problem05/data/train.csv"); te = pd.read_csv("problem05/data/test.csv"); F = ["pushup", "run_time", "sleep_quality", "stress_level"]
X = t[F].values.astype(float); fit = (t.status == "fit").values; rec = t.recovery_days.values.astype(float); SIGN = np.array([1, -1, 1, -1]); SC = 2.46475151; R = 20
STEP = np.array([1.0, 0.1, 0.1, 0.1])            # 각 특징의 기록 단위 (pushup 정수, 나머지 소수 1자리)


def interval(Xa, ya):
    """부호를 맞춘 좌표(V = X*SIGN, 클수록 fit)에서 임계값 허용 구간 (lo, hi]. hi = fit 극값, lo = 그 기준만 미달한 unfit 중 가장 가까운 값"""
    V = Xa * SIGN; hi = V[ya].min(0); lo = np.full(4, -np.inf)
    for j in range(4):
        others = np.all(np.delete(V, j, 1) >= np.delete(hi, j), axis=1); below = V[(~ya) & others & (V[:, j] < hi[j]), j]
        if len(below): lo[j] = below.max()
    return lo, hi


class Box:                                           # 메인: fit 극값
    def fit(self, Xa, ya): self.th = interval(Xa, ya)[1]; return self
    def predict(self, Xb): return ((Xb * SIGN) >= self.th - 1e-9).all(1)


class LooseBox(Box):                                 # 가장 가까운 unfit 바로 안쪽까지 넓힘 (그런 unfit 이 없으면 극값 유지)
    def fit(self, Xa, ya): lo, hi = interval(Xa, ya); self.th = np.where(np.isfinite(lo), lo + STEP, hi); return self


class MidBox(Box):                                   # 극값과 가장 가까운 unfit 의 중간
    def fit(self, Xa, ya): lo, hi = interval(Xa, ya); self.th = np.where(np.isfinite(lo), (lo + hi) / 2, hi); return self


class EVBox(Box):
    """극값 편향 보정: fit 값들의 평균 간격의 k 배만큼 바깥으로. 단, 가장 가까운 unfit 은 넘지 않음"""
    def __init__(self, k=1.0): self.k = k
    def fit(self, Xa, ya):
        lo, hi = interval(Xa, ya); V = (Xa * SIGN)[ya]; gap = (V.max(0) - V.min(0)) / max(len(V) - 1, 1); th = hi - self.k * gap
        self.th = np.where(np.isfinite(lo), np.maximum(th, lo + STEP), th); return self


class RoundBox(Box):
    """허용 구간 (lo, hi] 안에서 가장 '깔끔한' 수를 임계값으로. 단위 10 -> 5 -> 1 -> 0.5 순으로 찾고, 같은 단위에 여러 개면 가장 느슨한 값. 없으면 mode 에 따라 극값 또는 가장 느슨한 값"""
    def __init__(self, fallback="loose", grids=(10, 5, 1, 0.5)): self.fallback, self.grids = fallback, grids
    def fit(self, Xa, ya):
        lo, hi = interval(Xa, ya); th = hi.copy()
        for j in range(4):
            if not np.isfinite(lo[j]): lo_j = hi[j] - (10 if j == 0 else 1.5)      # 근거가 되는 unfit 이 없으면 한정된 폭 안에서만 찾는다
            else: lo_j = lo[j]
            found = None
            for g in self.grids:
                c = np.arange(np.ceil((lo_j + 1e-9) / g), np.floor((hi[j] + 1e-9) / g) + 1) * g; c = c[(c > lo_j + 1e-9) & (c <= hi[j] + 1e-9)]
                if len(c): found = c.min(); break                                   # V 좌표에서 가장 작은 값 = 가장 느슨한 기준
            if found is None: found = (lo_j + STEP[j]) if (self.fallback == "loose" and np.isfinite(lo[j])) else hi[j]
            th[j] = found
        self.th = th; return self


class Dominance:
    """구조 가정 없이 단조성만 사용: 어떤 학습 fit 보다 네 기준 모두 같거나 좋으면 fit"""
    def fit(self, Xa, ya): self.P = (Xa * SIGN)[ya]; return self
    def predict(self, Xb): return np.array([np.any(np.all(v >= self.P - 1e-9, axis=1)) for v in Xb * SIGN])


class SoftAnd:
    """P(fit) = 네 기준별 시그모이드의 곱. 임계값 4개 + 공통 기울기 1개를 최대우도로 추정"""
    def fit(self, Xa, ya):
        self.m, self.s = Xa.mean(0), Xa.std(0); Z = (Xa - self.m) / self.s * SIGN; y = ya.astype(float); t0 = Z[ya].min(0) - 0.2
        def nll(w):
            a = np.exp(w[4]); lp = -np.logaddexp(0, -a * (Z - w[:4])).sum(1); return -(y * lp + (1 - y) * np.log1p(-np.exp(np.minimum(lp, -1e-9)))).sum() + 0.01 * a ** 2
        r = minimize(nll, np.r_[t0, np.log(5.0)], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6}); self.w = r.x; return self
    def predict(self, Xb):
        Z = (Xb - self.m) / self.s * SIGN; a = np.exp(self.w[4]); return np.exp(-np.logaddexp(0, -a * (Z - self.w[:4])).sum(1)) > 0.5


MODELS = {"[메인] 상자: fit 극값": Box, "상자: 가장 가까운 unfit 직전까지": LooseBox, "상자: 극값과 unfit 의 중간": MidBox, "상자: 극값 편향 보정 k=0.5": lambda: EVBox(0.5), "상자: 극값 편향 보정 k=1": lambda: EVBox(1.0),
          "상자: 깔끔한 수 (없으면 가장 느슨)": lambda: RoundBox("loose"), "상자: 깔끔한 수 (없으면 극값)": lambda: RoundBox("tight"), "상자: 깔끔한 수, 정수만": lambda: RoundBox("tight", grids=(10, 5, 1)),
          "지배 관계 규칙": Dominance, "곱 형태 로지스틱(Soft-AND)": SoftAnd,
          "단조 제약 부스팅": lambda: HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=1, max_iter=200, learning_rate=0.1, monotonic_cst=[1, -1, 1, -1], random_state=0),
          "로지스틱 C=1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))}

if __name__ == "__main__":
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0).split(X, fit)); res = {}
    print("== status 모델 비교: Stratified 5-Fold x 20 + 한 명씩 빼고 검증(80회) ==")
    print("  %-34s %-7s %-8s %-8s %-7s %s" % ("모델", "정확도", "놓친fit", "거짓fit", "점수", "LOO 정답(놓친/거짓)"))
    for n, mk in MODELS.items():
        acc, fn, fp, sc = [], [], [], []
        for r in range(R):
            ps = np.zeros(80, bool); oof = np.zeros(80)
            for a, b in cv[r * 5:(r + 1) * 5]:
                p = np.asarray(mk().fit(X[a], fit[a]).predict(X[b])).astype(bool); ps[b] = p; oof[b] = np.where(p, rec[a][fit[a]].mean(), rec[a][~fit[a]].mean())
            acc.append((ps == fit).mean()); fn.append((fit & ~ps).sum()); fp.append((~fit & ps).sum()); sc.append(50 * acc[-1] + 50 / (1 + np.sqrt(np.mean((rec - oof) ** 2)) / SC))
        pl = np.zeros(80, bool)
        for a, b in LeaveOneOut().split(X): pl[b] = np.asarray(mk().fit(X[a], fit[a]).predict(X[b])).astype(bool)
        res[n] = (np.array(acc), np.array(sc)); print("  %-34s %.4f  %-8.2f %-8.2f %.2f   %d/80 (%d/%d)" % (n, np.mean(acc), np.mean(fn), np.mean(fp), np.mean(sc), (pl == fit).sum(), (fit & ~pl).sum(), (~fit & pl).sum()), flush=True)
    b = res["[메인] 상자: fit 극값"][1]
    print("\n  메인 대비 점수 차 (20회 반복):")
    for n, (a, s) in res.items():
        if n.startswith("[메인]"): continue
        d = s - b; print("    %-34s %+.2f (sd %.2f, %2d/20 반복에서 더 높음)" % (n, d.mean(), d.std(), int((d > 0).sum())))
    print("\n== 전체 학습으로 정한 임계값과 테스트 예측 ==")
    Xt = te[F].values.astype(float)
    for n in ("[메인] 상자: fit 극값", "상자: 가장 가까운 unfit 직전까지", "상자: 깔끔한 수 (없으면 가장 느슨)", "상자: 깔끔한 수 (없으면 극값)", "상자: 깔끔한 수, 정수만"):
        m = MODELS[n]().fit(X, fit); p = m.predict(Xt); print("  %-34s pushup>=%.0f run<=%.1f sleep>=%.1f stress<=%.1f | 학습 정확도 %.4f | 테스트 fit: %s" % (n, *(m.th * SIGN), (m.predict(X) == fit).mean(), te.soldier_id[p].tolist()))
