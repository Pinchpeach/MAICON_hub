# 07_small_tabular.py  ── 소규모 정형(학습 약 100행 이하): 타깃 구조에 맞는 단순 모델 하나 + 5-Fold 반복 쌍 비교
# 사용 흐름 (먼저 02_inspect.py 의 target_structure / box_check 로 구조를 본다):
#   a = repeated_cv(LinearRegression, X, y)                    # 기준: 선형 (X: 수치형 DataFrame, y: Series)
#   b = repeated_cv(lambda: Tobit(lower=30), X, y)             # 타깃이 하한에 몰림 -> 검열 회귀
#   paired({'선형': a, 'Tobit': b})                             # 반복별 쌍 비교(몇 번 중 몇 번 낫나)
#   pred = Tobit(lower=30).fit(X, y).predict(X_test)
#   계수 합이 1 에 가까움 -> WeightedMean()        "모두 통과" 형태 -> BoxRule() + group_value()
#   residual_r2(model, X, y)                                    # 잔차를 RF 로 예측한 R2 가 0 이하면 남은 구조 없음 -> 중단
# 팁: 실전에서는 구조 모델 하나에서 멈춘다(연습 3문제 모두 이후 반복의 이득 0.5점 이하). 트리·부스팅·신경망은 80행에서 선형·규칙보다 나빴다.
#     계수 반올림(grid=)은 연습에서만 쓴다(격자를 잘못 고르면 -0.5 ~ -2.6점). 극값 임계값은 rate_check 로 test 양성 비율을 꼭 본다.
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor

def repeated_splits(y, task='reg', repeats=20, folds=5):
    """반복마다 (학습, 검증) 인덱스 쌍의 목록을 돌려준다. 회귀는 KFold(seed=반복 번호), 분류는 층화 분할."""
    n = len(y); idx = np.arange(n)
    if task == 'reg':
        return [list(KFold(folds, shuffle=True, random_state=r).split(idx)) for r in range(repeats)]
    sp = list(RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=0).split(idx, np.asarray(y)))
    return [sp[r * folds:(r + 1) * folds] for r in range(repeats)]

def repeated_cv(make, X, y, task='reg', repeats=20, folds=5, post=None):
    """5-Fold 를 repeats 번 반복한 점수 배열. 회귀는 RMSE(낮을수록 좋음), 분류는 정확도. make() 는 새 모델을 돌려준다."""
    X = pd.DataFrame(X).reset_index(drop=True); y = np.asarray(y); out = []
    for splits in repeated_splits(y, task, repeats, folds):
        oof = np.empty(len(y), dtype=float if task == 'reg' else object)
        for a, b in splits:
            p = make().fit(X.iloc[a], y[a]).predict(X.iloc[b]); oof[b] = post(p) if post else p
        out.append(np.sqrt(np.mean((y.astype(float) - oof) ** 2)) if task == 'reg' else np.mean(oof == y))
    return np.array(out, dtype=float)

def paired(results, lower_is_better=True):
    """results = {이름: repeated_cv 결과}. 첫 항목을 기준으로 평균·반복 표준편차와 몇 번 중 몇 번 나은지를 출력."""
    names = list(results); base = results[names[0]]
    for n in names:
        v = results[n]; win = int(((v < base) if lower_is_better else (v > base)).sum())
        tail = '' if n == names[0] else f' | 기준보다 좋은 반복 {win}/{len(v)}'
        print(f'  {n:24s} {v.mean():.4f} (반복 sd {v.std():.4f}){tail}')

class Tobit:
    """하한 lower·상한 upper 에서 검열된 선형 회귀(최대우도). 예측 = E[clip(mu + eps, lower, upper)].
    둘 다 None 이면 보통의 선형 회귀와 같다. grid 는 기울기 반올림 단위(연습 전용, 스칼라 또는 특징별 배열)."""
    def __init__(self, lower=None, upper=None, grid=None):
        self.lower, self.upper, self.grid = lower, upper, grid
    def _nll(self, mu, logsg, y, lo, hi):
        sg = np.exp(logsg); mid = ~(lo | hi)
        ll = norm.logpdf((y[mid] - mu[mid]) / sg).sum() - mid.sum() * logsg
        if lo.any(): ll += norm.logcdf((self.lower - mu[lo]) / sg).sum()
        if hi.any(): ll += norm.logcdf((mu[hi] - self.upper) / sg).sum()
        return -ll
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); m, s = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - m) / s]
        lo = (y <= self.lower + 1e-9) if self.lower is not None else np.zeros(len(y), bool)
        hi = (y >= self.upper - 1e-9) if self.upper is not None else np.zeros(len(y), bool)
        b0 = np.linalg.lstsq(Z, y, rcond=None)[0]; s0 = np.log(max(np.std(y - Z @ b0), 1e-2))
        r = minimize(lambda w: self._nll(Z @ w[:-1], w[-1], y, lo, hi), np.r_[b0, s0], method='BFGS')
        self.coef_ = r.x[1:-1] / s; self.intercept_ = float(r.x[0] - (self.coef_ * m).sum()); self.sigma_ = float(np.exp(r.x[-1]))
        self.n_censored_ = (int(lo.sum()), int(hi.sum()))
        if self.grid is not None:                                   # 연습 전용: 기울기를 깔끔한 값으로 반올림하고 절편·sigma 만 다시 추정
            g = np.asarray(self.grid, float); self.coef_ = np.round(self.coef_ / g) * g; lin = X @ self.coef_
            r = minimize(lambda w: self._nll(w[0] + lin, w[1], y, lo, hi), [np.mean(y - lin), np.log(self.sigma_)], method='BFGS')
            self.intercept_, self.sigma_ = float(r.x[0]), float(np.exp(r.x[1]))
        return self
    def predict(self, X):
        mu = self.intercept_ + np.asarray(X, float) @ self.coef_; sg = self.sigma_
        Fa = fa = 0.0; Fb = 1.0; fb = 0.0; out = 0.0
        if self.lower is not None:
            a = (self.lower - mu) / sg; Fa, fa = norm.cdf(a), norm.pdf(a); out = out + self.lower * Fa
        if self.upper is not None:
            b = (self.upper - mu) / sg; Fb, fb = norm.cdf(b), norm.pdf(b); out = out + self.upper * (1 - Fb)
        return out + mu * (Fb - Fa) - sg * (fb - fa)

class WeightedMean:
    """입력 항목들의 가중 평균(계수 합 1, 절편 없음). grid 는 가중치 반올림 단위(연습 전용)."""
    def __init__(self, grid=None): self.grid = grid
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        w = np.linalg.lstsq(X[:, :-1] - X[:, [-1]], y - X[:, -1], rcond=None)[0]; w = np.r_[w, 1 - w.sum()]
        if self.grid is not None:
            w = np.round(w / self.grid) * self.grid; w[np.argmax(w)] += 1 - w.sum()
        self.w_ = w; return self
    def predict(self, X): return np.asarray(X, float) @ self.w_

class BoxRule:
    """모든 기준을 통과하면 양성. 임계값 = 학습 양성의 극값. sign 은 특징별 방향(+1: 클수록 양성, -1: 작을수록 양성), None 이면 평균 비교로 정한다."""
    def __init__(self, sign=None): self.sign = sign
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y).astype(bool)
        self.sign_ = np.asarray(self.sign, float) if self.sign is not None else np.where(X[y].mean(0) >= X[~y].mean(0), 1.0, -1.0)
        self.th_ = (X * self.sign_)[y].min(0); return self
    def predict(self, X): return ((np.asarray(X, float) * self.sign_) >= self.th_).all(1)
    def report(self, X, y, names=None):
        """임계값, 상자 안의 음성 수, 임계값을 정한 양성 행 수(의존도)를 출력."""
        X = np.asarray(X, float); y = np.asarray(y).astype(bool); Z = X * self.sign_
        names = list(names) if names is not None else [f'x{j}' for j in range(X.shape[1])]
        at = (Z[y] <= self.th_ + 1e-12)                              # 양성 중 임계값에 걸린 행
        for j, n in enumerate(names):
            op = '>=' if self.sign_[j] > 0 else '<='
            print(f'  {n} {op} {self.th_[j] * self.sign_[j]:g}  (이 값을 정한 양성 {int(at[:, j].sum())}명)')
        print(f'  상자 안 음성 {int(self.predict(X)[~y].sum())}명 / 음성 {int((~y).sum())}명 | 임계값을 하나라도 정한 양성 {int(at.any(1).sum())}명 / 양성 {int(y.sum())}명')

def group_value(pred, value_train, y_train, how='median'):
    """예측 클래스별로 학습 데이터의 값(예: 회복일수)을 채운다. 소규모에서는 평균보다 중앙값(특이값에 둔감)."""
    v = np.asarray(value_train, float); y = np.asarray(y_train).astype(bool); f = np.median if how == 'median' else np.mean
    return np.where(np.asarray(pred).astype(bool), f(v[y]), f(v[~y]))

def rate_check(pred_test, y_train):
    """극값 규칙은 양성을 놓치는 쪽으로 치우친다. test 양성 예측 비율이 train 보다 크게 낮으면 위험으로 보고한다(13.4)."""
    p = np.asarray(pred_test).astype(bool); y = np.asarray(y_train).astype(bool)
    print(f'  양성 비율: train {y.mean():.3f} | test 예측 {p.mean():.3f} ({int(p.sum())}/{len(p)}, train 비율이면 약 {y.mean() * len(p):.1f}명)')
    if p.mean() < 0.5 * y.mean(): print('  경고: test 양성 예측 비율이 train 의 절반 미만 -> 임계값이 너무 엄격할 수 있음(보고할 것)')

def residual_r2(model, X, y, folds=5, seed=0, n_jobs=2):
    """적합한 모델의 잔차를 RF 로 교차 예측한 R2. 0 이하면 입력으로 설명할 구조가 남지 않았다(개선 중단 근거)."""
    res = np.asarray(y, float) - model.predict(X)
    p = cross_val_predict(RandomForestRegressor(300, min_samples_leaf=3, random_state=seed, n_jobs=n_jobs), np.asarray(X, float), res,
                          cv=KFold(folds, shuffle=True, random_state=seed))
    return float(1 - np.sum((res - p) ** 2) / np.sum((res - res.mean()) ** 2))
