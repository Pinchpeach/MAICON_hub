# solution_new/experiments 폴더에서 실행 (상대 경로). p2_resnn.py 는 p2_clean.py 가 만든 p2_clean_store.npy 가 필요함
import time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.base import BaseEstimator, ClassifierMixin
SP = r'C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-root-docker11\7ec5aa84-89af-4967-a0fe-dd8714961a0f\scratchpad\\'
tr = pd.read_csv(r'..\data\train.csv'); X = tr.review.values; y = tr.label.values
class NBLR(BaseEstimator, ClassifierMixin):
    def __init__(self, C=8, alpha=1.0): self.C = C; self.alpha = alpha
    def _vec(self): return [CountVectorizer(analyzer='char_wb', ngram_range=(1, 4), binary=True, min_df=2), CountVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, token_pattern=r'\S+', min_df=2)]
    def fit(self, T, y):
        self.v_ = [v.fit(T) for v in self._vec()]; M = hstack([v.transform(T) for v in self.v_]).tocsr()
        p = self.alpha + M[y == 1].sum(0); q = self.alpha + M[y == 0].sum(0); self.r_ = np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()
        self.lr_ = LogisticRegression(C=self.C, max_iter=3000).fit(M.multiply(self.r_).tocsr(), y); self.classes_ = self.lr_.classes_; return self
    def predict_proba(self, T): return self.lr_.predict_proba(hstack([v.transform(T) for v in self.v_]).tocsr().multiply(self.r_).tocsr())
build = lambda: [make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True, min_df=2), ComplementNB(alpha=0.5)),
                 make_pipeline(CountVectorizer(analyzer='char_wb', ngram_range=(1, 4), binary=True, min_df=2), MultinomialNB(alpha=0.5)), NBLR(C=8)]
rank = lambda p: pd.Series(p).rank().values / len(p)
def trio_probs(Tt, Ty, At): return [m.fit(Tt, Ty).predict_proba(At)[:, 1] for m in build()]
def score_of(probs): return np.mean([rank(p) for p in probs], axis=0)           # 순위 평균(0~1)
lab = lambda s_, pr: (s_ >= np.quantile(s_, 1 - pr)).astype(int)
THS = (0.2, 0.3)
acc = {'base': np.zeros(len(y), dtype=int)}; acc.update({f'clean t={t}': np.zeros(len(y), dtype=int) for t in THS})
removed = {t: 0 for t in THS}; store = {}
t0 = time.time()
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    Ta, Ya = X[a], y[a]; pr = Ya.mean()
    # 학습 부분(a) 안에서 내부 5-Fold OOF -> 노이즈 의심 행 탐지
    inner = [np.zeros(len(a)) for _ in range(3)]
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=11).split(Ta, Ya):
        for j, p in enumerate(trio_probs(Ta[ia], Ya[ia], Ta[ib])): inner[j][ib] = p
    s_in = score_of(inner); wrong_in = lab(s_in, pr) != Ya; conf_in = np.abs(s_in - .5)
    sv = score_of(trio_probs(Ta, Ya, X[b])); acc['base'][b] = (lab(sv, pr) == y[b]).astype(int)
    store[k] = dict(a=a, b=b, s_in=s_in, s_val=sv)
    for t in THS:
        drop = wrong_in & (conf_in >= t); removed[t] += int(drop.sum()); keep = ~drop
        svc = score_of(trio_probs(Ta[keep], Ya[keep], X[b])); acc[f'clean t={t}'][b] = (lab(svc, pr) == y[b]).astype(int)
    print(f'fold {k}: base {acc["base"][b].mean():.4f} | ' + ' | '.join(f'clean t={t} {acc[f"clean t={t}"][b].mean():.4f} (제거 {int((wrong_in & (conf_in >= t)).sum())}행)' for t in THS) + f' | {time.time() - t0:.0f}s', flush=True)
np.save('p2_clean_store.npy', store, allow_pickle=True)
base = acc['base']
print(f'\n전체 CV 정확도: base {base.mean():.4f}')
rng = np.random.default_rng(0)
for t in THS:
    c = acc[f'clean t={t}']; d = []
    for _ in range(2000): i = rng.integers(0, len(y), len(y)); d.append(c[i].mean() - base[i].mean())
    print(f'  clean t={t}: {c.mean():.4f} (차이 {(c.mean() - base.mean()) * 100:+.3f}%p, 부트스트랩 95% [{np.percentile(d, 2.5) * 100:+.2f}, {np.percentile(d, 97.5) * 100:+.2f}]%p) | 평균 제거 행 {removed[t] / 5:.0f}개/fold ({removed[t] / 5 / 16000 * 100:.1f}%)')
