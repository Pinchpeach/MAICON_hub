# 09_sensor_window.py  ── 센서 시계열 분류(개체 단위 검증): 개체 안의 현재·과거 행만 쓰는 창 특징 + 평탄 RF·HGB 확률 평균
# 사용 흐름 (규칙 14.3 "센서 시계열 분류"):
#   df = df.sort_values(['uid', 'seq']).reset_index(drop=True)             # 반드시 개체·시간 순으로 정렬(train, test 각각)
#   df = add_norms(df, {'ang_norm': ['angular_x', 'angular_y', 'angular_z']})   # 벡터 크기 같은 파생 신호
#   F = window_features(df, 'uid', signals=['ang_norm', ...], slope_cols=['battery_current'], rel_cols=['battery_voltage'])
#   P, classes, parts = group_oof(F, y, df['uid'])                         # 개체 단위 OOF 확률(RF + HGB 평균), 모델별 소요 시간 출력
#   report(y, P, classes)                                                  # 정확도·Macro F1·클래스별 F1
#   boundary_share(y, classes[P.argmax(1)], df['uid'])                     # 전환 경계 근처 오류 비중(정확도 상한 진단)
#   Pt = fit_predict_proba(F, y, F_test, classes)                          # 전체 train 으로 다시 학습해 test 확률
# 팁: 기준선은 평탄(단일 단계) 모델이다. 계층 분류는 같은 특징에서 평탄보다 높지 않았다(Macro F1 0.736 대 0.747, 비행 5개).
#     원값 센서(전압·전류·바람)는 개체 간 분포 이동 때문에 넣으면 나빠졌다 -> slope_cols / rel_cols 로 변화량만 쓰고, 넣고 빼서 확인한다.
#     예측을 시간 방향으로 평활하면 전환이 뭉개져 해로웠다. 클래스 가중(report 의 weights)은 OOF 로 고르면 낙관적이라고 표기한다.
import time, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

def add_norms(df, groups):
    """groups = {새 열 이름: [성분 열들]} 의 벡터 크기 열을 더한다(각속도·가속도 등)."""
    df = df.copy()
    for name, cols in groups.items(): df[name] = np.sqrt((df[list(cols)].astype(float) ** 2).sum(axis=1))
    return df

def window_features(df, group, signals, windows=(5, 10, 50, 100, 200), spread=(10, 50), ewm_spans=(5, 20, 100),
                    slope_cols=(), slope_lags=(10, 50, 100), rel_cols=(), abs_cols=()):
    """group 안에서 현재·과거 행만 쓰는 특징(미래 값 미사용). df 는 group·시간 순으로 정렬되어 있어야 한다.
    signals: 창 평균(windows), 표준편차·최대·최소(spread), 지수이동평균(ewm_spans), 추세(짧은 창 평균 - 긴 창 평균)
    slope_cols: (현재 - n행 전) / n   rel_cols: 개체의 첫 값 대비 차이   abs_cols: 현재 행의 절대값"""
    uid = df[group].astype(str); F = pd.DataFrame(index=df.index)
    assert (uid != uid.shift()).sum() == uid.nunique(), '같은 개체의 행이 붙어 있지 않다: group·시간 순으로 정렬할 것'
    def roll(s, n, fn): return getattr(s.groupby(uid, sort=False).rolling(n, min_periods=2), fn)().reset_index(level=0, drop=True).sort_index()
    def ewm(s, span): return s.groupby(uid, sort=False).transform(lambda x: x.ewm(span=span, adjust=False).mean())
    for c in abs_cols: F[f'{c}_abs'] = df[c].abs()
    for c in signals:
        s = df[c].astype(float); F[c] = s
        for n in windows: F[f'{c}_m{n}'] = roll(s, n, 'mean')
        for n in spread:
            F[f'{c}_s{n}'] = roll(s, n, 'std'); F[f'{c}_max{n}'] = roll(s, n, 'max'); F[f'{c}_min{n}'] = roll(s, n, 'min')
        for sp in ewm_spans: F[f'{c}_e{sp}'] = ewm(s, sp)
        if len(windows) >= 3: F[f'{c}_trend'] = F[f'{c}_m{windows[1]}'] - F[f'{c}_m{windows[2]}']
    for c in slope_cols:
        s = df[c].astype(float); g = s.groupby(uid, sort=False)
        for n in slope_lags: F[f'{c}_slope{n}'] = (s - g.shift(n)) / n
    for c in rel_cols:
        s = df[c].astype(float); F[f'{c}_rel'] = s - s.groupby(uid, sort=False).transform('first')
    return F.fillna(0.0)

def flat_models(n_jobs=2, seed=0):
    """평탄 기준선 두 모델. HGB 는 sklearn 1.1.3 에 class_weight 가 없으므로 가중이 필요하면 sample_weight 를 쓴다."""
    return {'rf': RandomForestClassifier(300, min_samples_leaf=3, n_jobs=n_jobs, random_state=seed),
            'hgb': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, random_state=seed)}

def _proba(m, X, classes):
    """모델의 확률을 classes 순서의 열로 맞춘다(학습 폴드에 없는 클래스는 0)."""
    P = np.zeros((len(X), len(classes))); p = m.predict_proba(X)
    for j, c in enumerate(m.classes_): P[:, int(np.where(classes == c)[0][0])] = p[:, j]
    return P

def group_oof(F, y, groups, models=None, max_logo=10, n_jobs=2):
    """개체 단위 OOF 확률. 개체가 max_logo 개 이하면 하나씩 빼는 검증, 많으면 GroupKFold(5). 반환: (평균 확률, 클래스 배열, 모델별 확률)."""
    X = np.asarray(F, float); y = np.asarray(y); g = np.asarray(groups); classes = np.unique(y); models = models or flat_models(n_jobs)
    ug = pd.unique(g)
    splits = [(np.where(g != u)[0], np.where(g == u)[0]) for u in ug] if len(ug) <= max_logo else list(GroupKFold(5).split(X, y, g))
    parts = {k: np.zeros((len(X), len(classes))) for k in models}; sec = {k: 0.0 for k in models}
    for a, b in splits:
        for k, m in models.items():
            t0 = time.time(); parts[k][b] = _proba(clone(m).fit(X[a], y[a]), X[b], classes); sec[k] += time.time() - t0
    print(f'개체 {len(ug)}개, 폴드 {len(splits)}개, 행 {len(X)}, 특징 {X.shape[1]}개 | 소요 ' + ', '.join(f'{k} {v:.0f}초' for k, v in sec.items()))
    return np.mean(list(parts.values()), axis=0), classes, parts

def fit_predict_proba(F, y, F_test, classes=None, models=None, n_jobs=2):
    """전체 train 으로 다시 학습해 test 확률(모델 평균)을 돌려준다. 열 순서는 classes."""
    X = np.asarray(F, float); y = np.asarray(y); classes = np.unique(y) if classes is None else np.asarray(classes); models = models or flat_models(n_jobs)
    return np.mean([_proba(clone(m).fit(X, y), np.asarray(F_test, float), classes) for m in models.values()], axis=0)

def report(y, P, classes, weights=None, name=''):
    """정확도, Macro F1, 클래스별 F1. weights 는 클래스별 확률 가중(OOF 로 고른 값이면 낙관적이라고 표기)."""
    y = np.asarray(y); w = np.ones(len(classes)) if weights is None else np.asarray(weights, float)
    pred = np.asarray(classes)[(P * w).argmax(1)]; f = f1_score(y, pred, labels=list(classes), average=None, zero_division=0)
    print(f'  {name:16s} 정확도 {np.mean(pred == y):.4f} | Macro F1 {f.mean():.4f} | ' + ' '.join(f'{str(c)[:5]}:{v:.3f}' for c, v in zip(classes, f)))
    return float(np.mean(pred == y)), float(f.mean())

def boundary_share(y, pred, groups, k=5):
    """라벨 전환 경계에서 k행 이내인 행의 오류 비중과 짧은 구간 비율. 경계 오류가 대부분이면 센서만으로는 상한이 낮다(목표를 먼저 알린다)."""
    y = np.asarray(y); pred = np.asarray(pred); g = np.asarray(groups); n = len(y)
    start = np.r_[True, (y[1:] != y[:-1]) | (g[1:] != g[:-1])]; seg = np.cumsum(start) - 1; seg_len = np.bincount(seg)
    change = np.r_[False, (y[1:] != y[:-1]) & (g[1:] == g[:-1])]; idx = np.where(change)[0]; near = np.zeros(n, bool)
    for i in idx: near[max(0, i - k):i + k] = True
    err = pred != y; acc = 1 - err.mean()
    print(f'  구간 {len(seg_len)}개 중 1~2행짜리 {np.mean(seg_len <= 2):.1%} | 경계 {k}행 이내 행 {near.mean():.1%}')
    print(f'  전체 정확도 {acc:.4f} | 경계 근처 정확도 {1 - err[near].mean() if near.any() else float("nan"):.4f} | 그 밖 {1 - err[~near].mean():.4f}')
    print(f'  오류 중 경계 근처 비중 {err[near].sum() / max(err.sum(), 1):.1%} | 경계 근처 오류를 모두 없앤 가상 정확도 {1 - err[~near].sum() / n:.4f}')
    return float(err[near].sum() / max(err.sum(), 1))
