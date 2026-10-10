# 03_tabular.py  ── 표 데이터 기준 구성: 부스팅 트리(HGB) + Random Forest, K-Fold CV, 로그 타깃 지원
# 사용 흐름:
#   X['cut'] = clean_ordinal(X['cut'], ['Fair','Good','Very Good','Premium','Ideal'])   # 표기 정리 + 순서형 인코딩
#   models = tabular_models('reg')                       # 분류는 'clf'
#   oof, scores = run_cv(models, X, y, 'reg', folds=3, log_target=True)
#   pred = fit_full_predict(models, X, y, X_test, 'reg', log_target=True)
# 팁: 학습이 5분 넘으면 folds=3 이나 부분 데이터로 비교, 모델 교체보다 정리·결측 복원·파생 특징·타깃 변환이 먼저.
import time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import (HistGradientBoostingRegressor, HistGradientBoostingClassifier,
                              RandomForestRegressor, RandomForestClassifier)
warnings.filterwarnings('ignore')

def clean_ordinal(s, order):
    """문자열 열의 공백·대소문자를 정리하고 order(낮음->높음) 순서의 정수로 인코딩. 매핑에 없으면 NaN."""
    m = {str(o).strip().lower(): i for i, o in enumerate(order)}
    return s.astype(str).str.strip().str.lower().map(m)

def tabular_models(task='reg', seed=0):
    """기준 구성: 부스팅 트리 + Random Forest (RF 는 결측을 허용하지 않으므로 run_cv/fit_full_predict 가 중앙값으로 채움)."""
    if task == 'reg':
        return {'hgb': HistGradientBoostingRegressor(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=seed),
                'rf': RandomForestRegressor(300, min_samples_leaf=2, max_features=0.6, n_jobs=-1, random_state=seed)}
    return {'hgb': HistGradientBoostingClassifier(learning_rate=0.06, max_iter=400, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0, random_state=seed),
            'rf': RandomForestClassifier(300, min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=seed)}

def _fit_pred(name, model, Xa, ya, Xb, task, log_target, fill):
    if name == 'rf': Xa, Xb = Xa.fillna(fill), Xb.fillna(fill)
    from sklearn.base import clone
    m = clone(model)
    if task == 'reg':
        m.fit(Xa, np.log1p(ya) if log_target else ya); p = m.predict(Xb)
        return np.expm1(p) if log_target else p
    m.fit(Xa, ya); return m.predict_proba(Xb)

def _metric(task, y, p, log_target):
    if task == 'reg':                                  # RMSE (log_target 이면 log1p 공간 = RMSLE)
        a, b = (np.log1p(y), np.log1p(np.clip(p, 0, None))) if log_target else (y, p)
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return float((p.argmax(1) == pd.Series(y).astype('category').cat.codes.values).mean())

def run_cv(models, X, y, task='reg', folds=5, seed=42, log_target=False):
    """OOF 예측과 모델별/평균 점수를 반환. 반환: (oof dict, scores dict)"""
    y = np.asarray(y); fill = X.median(numeric_only=True)
    cv = (KFold if task == 'reg' else StratifiedKFold)(folds, shuffle=True, random_state=seed)
    oof = {k: (np.zeros(len(X)) if task == 'reg' else np.zeros((len(X), len(np.unique(y))))) for k in models}
    for k, m in models.items():
        t0 = time.time()
        for a, b in cv.split(X, y):
            oof[k][b] = _fit_pred(k, m, X.iloc[a], y[a], X.iloc[b], task, log_target, fill)
        print(f'  {k:5s} CV {_metric(task, y, oof[k], log_target):.4f}  ({time.time() - t0:.0f}s)')
    blend = np.mean(list(oof.values()), axis=0)
    scores = {k: _metric(task, y, v, log_target) for k, v in oof.items()}; scores['blend'] = _metric(task, y, blend, log_target)
    print(f'  평균 앙상블 CV {scores["blend"]:.4f}  ({"낮을수록" if task == "reg" else "높을수록"} 좋음)')
    return oof, scores

def fit_full_predict(models, X, y, X_test, task='reg', log_target=False, seeds=(0,)):
    """전체 train 으로 재학습해 test 예측(모델 평균, 시드 평균). 회귀는 실수, 분류는 확률 행렬을 반환."""
    y = np.asarray(y); fill = X.median(numeric_only=True); preds = []
    for s in seeds:
        for k, m in tabular_models(task, s).items():
            if k in models: preds.append(_fit_pred(k, m, X, y, X_test, task, log_target, fill))
    p = np.mean(preds, axis=0)
    return np.clip(p, 0, None) if (task == 'reg' and log_target) else p
