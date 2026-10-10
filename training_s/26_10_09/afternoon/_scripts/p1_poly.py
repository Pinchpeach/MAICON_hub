import sys, os, warnings, itertools; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from cv135b import Tobit

t = pd.read_csv("problem01/data/train.csv"); F = ["avg_sleep_hours", "night_shift_days", "caffeine_intake"]; X = t[F].values.astype(float); y = t.fatigue_score.values.astype(float); L = 30.0
S = lambda r: 100 / (1 + r / 9.65650938); R = 20


def poly(Xa, Xb):
    """2차 다항 특성(9개: 원 3 + 제곱 3 + 교차 3)을 만들고 학습 부분 기준으로 표준화"""
    pf = PolynomialFeatures(2, include_bias=False); A = pf.fit_transform(Xa); B = pf.transform(Xb); m, s = A.mean(0), A.std(0) + 1e-9
    return (A - m) / s, (B - m) / s


def fit_poly_ridge(Xa, ya, Xb, alpha):
    """A: 다항 Ridge. 학습은 보통 제곱오차, 예측만 하한 30 으로 자름"""
    A, B = poly(Xa, Xb); return np.clip(Ridge(alpha=alpha).fit(A, ya).predict(B), L, None)


def fit_cens(Z, Zb, ya, lam):
    """검열(하한 30) 우도 + L2 벌점. 반환: E[max(30, mu + eps)]"""
    Z1 = np.c_[np.ones(len(Z)), Z]; cen = ya <= L + 1e-9
    b0 = np.r_[ya.mean(), np.zeros(Z.shape[1])]; b0[1:] = np.linalg.lstsq(Z.T @ Z + (lam + 1e-6) * np.eye(Z.shape[1]), Z.T @ (ya - ya.mean()), rcond=None)[0]
    s0 = np.log(max(np.std(ya - Z1 @ b0), 1e-2))
    def nll(w):
        b, sg = w[:-1], np.exp(w[-1]); mu = Z1 @ b
        return -(norm.logpdf((ya[~cen] - mu[~cen]) / sg).sum() - (~cen).sum() * np.log(sg) + norm.logcdf((L - mu[cen]) / sg).sum()) + lam * (b[1:] ** 2).sum() / (2 * sg ** 2)
    r = minimize(nll, np.r_[b0, s0], method="BFGS"); b, sg = r.x[:-1], float(np.exp(r.x[-1])); mu = np.c_[np.ones(len(Zb)), Zb] @ b; z = (mu - L) / sg
    return L + (mu - L) * norm.cdf(z) + sg * norm.pdf(z)


def fit_poly_cens(Xa, ya, Xb, lam):
    """B: 다항 특성 + Ridge 벌점 + 30 하한을 학습에 반영"""
    A, B = poly(Xa, Xb); return fit_cens(A, B, ya, lam)


def fit_tobit_nice(Xa, ya, Xb, grid=np.array([1.0, 0.5, 0.05])):
    """현재 메인: 선형 Tobit 후 기울기 반올림, 절편·sigma 재추정"""
    m = Tobit().fit(Xa, ya); sl = np.round((m.b[1:] / m.s) / grid) * grid; lin = Xa @ sl; cen = ya <= L + 1e-9
    def nll(w):
        mu = w[0] + lin; sg = np.exp(w[1]); return -(norm.logpdf((ya[~cen] - mu[~cen]) / sg).sum() - (~cen).sum() * np.log(sg) + norm.logcdf((L - mu[cen]) / sg).sum())
    r = minimize(nll, [np.mean(ya - lin), np.log(m.sg)], method="BFGS"); mu = r.x[0] + Xb @ sl; sg = float(np.exp(r.x[1])); z = (mu - L) / sg
    return L + (mu - L) * norm.cdf(z) + sg * norm.pdf(z)


ALPHAS = (0.3, 1, 3, 10, 30, 100); LAMS = (0.3, 1, 3, 10, 30, 100)
MODELS = {"main": lambda a, ya, b: fit_tobit_nice(a, ya, b), "tobit": lambda a, ya, b: Tobit().fit(a, ya).predict(b)}
for al in ALPHAS: MODELS[f"A{al}"] = (lambda al: lambda a, ya, b: fit_poly_ridge(a, ya, b, al))(al)
for lm in LAMS: MODELS[f"B{lm}"] = (lambda lm: lambda a, ya, b: fit_poly_cens(a, ya, b, lm))(lm)
rm = lambda o: np.sqrt(np.mean((y - o) ** 2))

if __name__ == "__main__":
    OOF = {k: np.zeros((R, 80)) for k in MODELS}
    for r in range(R):
        for a, b in KFold(5, shuffle=True, random_state=r).split(X):
            for k, f in MODELS.items(): OOF[k][r, b] = f(X[a], y[a], X[b])
    sc = {k: np.array([rm(v[r]) for r in range(R)]) for k, v in OOF.items()}
    def show(name, v, ref="main"):
        d = v - sc[ref]; print(f"  {name:46s} RMSE {v.mean():.4f} (sd {v.std():.4f}) 점수 {S(v.mean()):.2f} | 메인 대비 {d.mean():+.4f}, {int((d < 0).sum()):2d}/{R} 반복에서 더 좋음")
    print("== 단일 모델 (5-Fold x 20) ==")
    show("현재 메인: 선형 Tobit 기울기 반올림", sc["main"]); show("선형 Tobit (v1)", sc["tobit"])
    print(" -- A: 2차 다항 + Ridge, 예측만 30 으로 자름 --")
    for al in ALPHAS: show(f"A alpha={al}", sc[f"A{al}"])
    print(" -- B: 2차 다항 + Ridge 벌점 + 30 하한을 학습에 반영 --")
    for lm in LAMS: show(f"B lam={lm}", sc[f"B{lm}"])
    bestA = min((f"A{al}" for al in ALPHAS), key=lambda k: sc[k].mean()); bestB = min((f"B{lm}" for lm in LAMS), key=lambda k: sc[k].mean())
    print(f"\n== 혼합 예측 (가중 w x 앞 모델 + (1-w) x 뒤 모델). 최선 A={bestA}, 최선 B={bestB} — 전체 검증으로 고른 값이라 낙관적 ==")
    for p, q in ((bestB, bestA), ("main", bestA), ("main", bestB), ("tobit", bestB)):
        for w in (0.25, 0.5, 0.75):
            v = np.array([rm(w * OOF[p][r] + (1 - w) * OOF[q][r]) for r in range(R)]); show(f"{w:.2f} x {p} + {1 - w:.2f} x {q}", v)
    # ---------- 중첩 교차검증: 벌점 세기와 혼합 가중을 안쪽 분할에서만 고름 ----------
    print("\n== 중첩 교차검증 (바깥 5-Fold x 10, 안쪽 4-Fold 로 alpha·lam·w 선택) ==")
    AG, LG, WG = (1, 10, 100), (1, 10, 100), (0, 0.25, 0.5, 0.75, 1.0); RN = 10; res = {k: [] for k in ("A 선택", "B 선택", "B+A 혼합 선택", "메인+B 혼합 선택", "메인")}; picks = []
    for r in range(RN):
        oo = {k: np.zeros(80) for k in res}
        for a, b in KFold(5, shuffle=True, random_state=r).split(X):
            Xa, ya = X[a], y[a]; inn = {**{f"A{al}": np.zeros(len(a)) for al in AG}, **{f"B{lm}": np.zeros(len(a)) for lm in LG}, "main": np.zeros(len(a))}
            for ia, ib in KFold(4, shuffle=True, random_state=100 + r).split(Xa):
                for al in AG: inn[f"A{al}"][ib] = fit_poly_ridge(Xa[ia], ya[ia], Xa[ib], al)
                for lm in LG: inn[f"B{lm}"][ib] = fit_poly_cens(Xa[ia], ya[ia], Xa[ib], lm)
                inn["main"][ib] = fit_tobit_nice(Xa[ia], ya[ia], Xa[ib])
            e = lambda o: np.sqrt(np.mean((ya - o) ** 2))
            ka = min((f"A{al}" for al in AG), key=lambda k: e(inn[k])); kb = min((f"B{lm}" for lm in LG), key=lambda k: e(inn[k]))
            w1 = min(WG, key=lambda w: e(w * inn[kb] + (1 - w) * inn[ka])); w2 = min(WG, key=lambda w: e(w * inn["main"] + (1 - w) * inn[kb]))
            pa = fit_poly_ridge(Xa, ya, X[b], float(ka[1:])); pb = fit_poly_cens(Xa, ya, X[b], float(kb[1:])); pm = fit_tobit_nice(Xa, ya, X[b])
            oo["A 선택"][b] = pa; oo["B 선택"][b] = pb; oo["B+A 혼합 선택"][b] = w1 * pb + (1 - w1) * pa; oo["메인+B 혼합 선택"][b] = w2 * pm + (1 - w2) * pb; oo["메인"][b] = pm; picks.append((ka, kb, w1, w2))
        for k in res: res[k].append(rm(oo[k]))
    for k, v in res.items():
        v = np.array(v); d = v - np.array(res["메인"]); print(f"  {k:22s} RMSE {v.mean():.4f} (sd {v.std():.4f}) 점수 {S(v.mean()):.2f} | 메인 대비 {d.mean():+.4f}, {int((d < 0).sum())}/{RN} 반복에서 더 좋음")
    P = pd.DataFrame(picks, columns=["A", "B", "w(B vs A)", "w(메인 vs B)"]); print("  안쪽에서 고른 값 빈도:", {c: P[c].value_counts().to_dict() for c in P})
