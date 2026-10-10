# 04_text.py  ── 작은 텍스트 분류(이진·다중): 글자 n-gram NB 와 글자 TF-IDF + 선형 SVM 을 둘 다 한 번씩 돌려 높은 쪽을 메인으로
# 사용 흐름 (규칙 14.3 "작은 텍스트 분류", 2026-10-10 개정):
#   oof, scores, best = cv_text(X_text, y, folds=5)                 # 모델별 CV 정확도·Macro F1·소요 시간. best = 가장 높은 단일 모델
#   pred, S = fit_text_predict(X_text, y, X_test_text, names=[best]) # 메인 = 단일 최고 모델. pred 는 원래 라벨, S 는 점수 행렬
#   (이진·임계값 규칙이 필요할 때) pred = rate_labels(binary_score(S), pos_rate)   # 비율 규칙은 정답을 보기 전에 정하고 기록
# 팁: 어느 쪽이 높은지는 문제마다 달랐다(영화 리뷰 감성은 NB 0.845 대 SVM 0.840, 뉴스 주제 7분류는 SVM 0.841 대 NB 0.82).
#     결합은 기본값이 아니다. 두 문제 모두 이득이 표준오차 안이거나 손해였다(감성 NB + SVM +0.1~0.2%p, 뉴스 주제 -0.1~-1.0%p).
#     cv_text 가 결합 후보를 함께 보여 주며, 단일 최고보다 표준오차의 2배 이상 높을 때만 names=[...] 에 여러 개를 넣는다.
#     다중 분류에서 클래스 열별 순위 평균은 쓰지 않는다(뉴스 주제 Macro F1 -1.0%p). 여기서는 점수 행렬 전체를 표준화해 평균한다.
#     글자 LR 은 같은 특징에서 SVM 보다 낮고 느렸다(3만 건 97초). 변형 간 차이가 표준오차 안이면 멈춘다(13.11).
import time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

def make_vec(ngram=(1, 4), min_df=2):
    return TfidfVectorizer(analyzer='char_wb', ngram_range=ngram, sublinear_tf=True, min_df=min_df)

def text_models(C=0.3, alpha=0.3):
    """같은 글자 TF-IDF 특징 위의 세 모델. 특징은 폴드마다 한 번만 만들어 함께 쓴다."""
    return {'cnb': ComplementNB(alpha=alpha), 'mnb': MultinomialNB(alpha=alpha), 'svm': LinearSVC(C=C)}

def _score_matrix(m, A):
    """(행, 클래스) 점수 행렬. NB 는 로그 확률, SVM 은 결정 함수(이진이면 두 열로 편다)."""
    if hasattr(m, 'predict_log_proba'): return m.predict_log_proba(A)
    d = m.decision_function(A)
    return np.c_[-d, d] if d.ndim == 1 else d

def combine(mats):
    """점수 행렬마다 전체 평균·표준편차로 표준화한 뒤 평균한다(모델마다 점수 크기가 달라도 클래스 간 비교가 유지된다)."""
    return np.mean([(m - m.mean()) / (m.std() + 1e-9) for m in mats], axis=0)

def binary_score(S): return S[:, 1] - S[:, 0]

def rate_labels(score, pos_rate):
    """점수 상위 pos_rate 비율을 1 로 정한다(클래스 비율이 train 과 비슷하다는 근거가 있을 때만)."""
    return (score >= np.quantile(score, 1 - pos_rate)).astype(int)

def _report(name, yi, S, sec=None):
    p = S.argmax(1); acc = float((p == yi).mean()); f1 = float(f1_score(yi, p, average='macro'))
    print(f'  {name:14s} 정확도 {acc:.4f} | Macro F1 {f1:.4f}' + (f' | {sec:.0f}초' if sec is not None else ''))
    return {'acc': acc, 'f1': f1, 'sec': sec}

def cv_text(X, y, models=None, folds=5, seed=42, ngram=(1, 4), combos=(('cnb', 'mnb'), ('cnb', 'svm')), metric=None):
    """모델별 OOF 점수 행렬과 CV 점수를 돌려준다. metric 은 'acc' 또는 'f1'(기본: 이진은 정확도, 다중은 Macro F1)."""
    X = np.asarray(X, dtype=object); classes, yi = np.unique(np.asarray(y), return_inverse=True); models = models or text_models()
    metric = metric or ('acc' if len(classes) == 2 else 'f1'); oof = {k: np.zeros((len(X), len(classes))) for k in models}; sec = {k: 0.0 for k in models}
    t0 = time.time(); tv = 0.0
    for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, yi):
        t1 = time.time(); v = make_vec(ngram); A, B = v.fit_transform(X[a]), v.transform(X[b]); tv += time.time() - t1
        for k, m in models.items():
            t1 = time.time(); oof[k][b] = _score_matrix(clone(m).fit(A, yi[a]), B); sec[k] += time.time() - t1
    print(f'클래스 {len(classes)}개, {len(X)}건, 특징 약 {A.shape[1]}개 | 벡터화 {tv:.0f}초 | 선택 지표 {metric}')
    scores = {k: _report(k, yi, oof[k], sec[k]) for k in models}
    best = max(models, key=lambda k: scores[k][metric]); b = scores[best][metric]; se = float(np.sqrt(b * (1 - b) / len(X)))
    for c in combos:
        if all(k in oof for k in c):
            s = _report('+'.join(c) + ' 결합', yi, combine([oof[k] for k in c])); scores['+'.join(c)] = s; d = s[metric] - b
            print(f'      -> 단일 최고 대비 {d:+.4f} (표준오차 약 {se:.4f}): ' + ('후보(2배 이상 높음)' if d > 2 * se else '쓰지 않음(노이즈 범위이거나 낮음)'))
    print(f'단일 최고: {best} ({metric} {scores[best][metric]:.4f}) | 전체 {time.time() - t0:.0f}초')
    return oof, scores, best

def fit_text_predict(X, y, X_test, names=('svm',), models=None, ngram=(1, 4)):
    """전체 train 으로 다시 학습해 test 를 예측. names 가 하나면 그 모델, 여러 개면 표준화 평균 결합. 반환: (원래 라벨 예측, 점수 행렬)."""
    X = np.asarray(X, dtype=object); X_test = np.asarray(X_test, dtype=object); classes, yi = np.unique(np.asarray(y), return_inverse=True)
    models = models or text_models(); v = make_vec(ngram); A, B = v.fit_transform(X), v.transform(X_test)
    mats = [_score_matrix(clone(models[k]).fit(A, yi), B) for k in names]
    S = mats[0] if len(mats) == 1 else combine(mats)
    return classes[S.argmax(1)], S
