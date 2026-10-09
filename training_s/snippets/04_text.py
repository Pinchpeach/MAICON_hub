# 04_text.py  ── 작은 텍스트 이진 분류: 글자 n-gram NB(순위 평균) + TF-IDF LR 을 함께 비교
# 사용 흐름:
#   models = text_models()
#   oof = cv_text(models, X_text, y, folds=5)                    # 모델별 CV 정확도 + 순위 평균
#   score = fit_text_predict(models, X_text, y, X_test_text)      # 순위 평균 점수
#   pred = rate_labels(score, y.mean())                          # 상위 비율을 1 로(클래스 비율이 비슷하다는 근거가 있을 때)
# 팁: 확률 평균은 NB 의 확률 보정 때문에 희석된다 -> 순위 평균. 임계값 규칙은 정답을 보기 전에 정하고 기록한다.
import time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

def text_models():
    word_char = lambda: make_union(TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True, min_df=2),
                                   TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, token_pattern=r'\S+', min_df=2))
    return {'cnb': make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True, min_df=2), ComplementNB(alpha=0.5)),
            'mnb': make_pipeline(CountVectorizer(analyzer='char_wb', ngram_range=(1, 4), binary=True, min_df=2), MultinomialNB(alpha=0.5)),
            'lr': make_pipeline(word_char(), LogisticRegression(C=4, max_iter=2000))}

rank = lambda p: pd.Series(p).rank().values / len(p)

def rate_labels(score, pos_rate):
    """점수 상위 pos_rate 비율을 1 로 정한다."""
    return (score >= np.quantile(score, 1 - pos_rate)).astype(int)

def cv_text(models, X, y, folds=5, seed=42):
    X = np.asarray(X, dtype=object); y = np.asarray(y); oof = {k: np.zeros(len(X)) for k in models}
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for k, m in models.items():
        t0 = time.time()
        for a, b in cv.split(X, y): oof[k][b] = clone(m).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        print(f'  {k:4s} CV acc {((oof[k] >= .5) == y).mean():.4f}  ({time.time() - t0:.0f}s)')
    ens = np.mean([rank(p) for p in oof.values()], axis=0)
    print(f'  순위 평균 CV acc {(rate_labels(ens, y.mean()) == y).mean():.4f}')
    return oof

def fit_text_predict(models, X, y, X_test):
    X = np.asarray(X, dtype=object); y = np.asarray(y); X_test = np.asarray(X_test, dtype=object)
    return np.mean([rank(clone(m).fit(X, y).predict_proba(X_test)[:, 1]) for m in models.values()], axis=0)
