# 02_inspect.py  (T+0~5, 약 2분)  ── 데이터 점검: 결측·표기 변형·0값·중복·id·타깃·분포 이동
# 사용: inspect_data(train, test, target='price', id_col='id', sample=sample_submission)
#       target_structure(train, 'fatigue_score', features)      # 공통 첫 단계(규칙 14.3): 타깃의 쏠림·기록 단위·합 제약을 모델 선택 전에 본다
#       box_check(train[features], train.status == 'fit')       # 이진 타깃이 '모두 통과' 규칙형인지
# 팁: 여기서 보인 구조(하한 쏠림, 계수 합 1, 상자 안 음성 0명)가 요약표의 '출제 의도 가설' 한 줄이 된다. 가설은 기준선 직후 실험 한 번으로 확인한다.
import numpy as np, pandas as pd

def inspect_data(train, test=None, target=None, id_col=None, sample=None):
    print('train', train.shape, '| test', None if test is None else test.shape)
    na = train.isna().sum(); print('결측(train):', na[na > 0].to_dict() or '없음')
    if test is not None:
        na2 = test.isna().sum(); print('결측(test):', na2[na2 > 0].to_dict() or '없음')
        print('train 에만 있는 열:', sorted(set(train.columns) - set(test.columns)),
              '| test 에만 있는 열:', sorted(set(test.columns) - set(train.columns)))
    for c in train.select_dtypes(exclude='number').columns:      # 문자열 열: 대소문자·공백 변형 탐지
        s = train[c].astype(str); raw_n, norm_n = s.nunique(), s.str.strip().str.lower().nunique()
        flag = '  <-- 표기 변형 의심(공백/대소문자)' if norm_n < raw_n else ''
        shown = s.value_counts().head(8).to_dict() if raw_n <= 30 else f'예시 {s.sample(3, random_state=0).tolist()}'
        print(f'[{c}] 고유값 {raw_n}개(정리 후 {norm_n}){flag}', shown)
    num = train.select_dtypes(include='number')
    if len(num.columns):
        print(num.describe().T[['min', '50%', 'max']].round(3).to_string())
        zeros = (num == 0).sum(); print('0 값이 있는 열:', zeros[zeros > 0].to_dict() or '없음')
    key = [c for c in train.columns if c != id_col]
    print('중복 행(id 제외):', int(train.duplicated(subset=key).sum()))
    if target is not None:
        t = train[target]
        print('타깃:', t.value_counts().sort_index().to_dict() if t.nunique() <= 20 else t.describe().round(3).to_dict())
    if id_col and test is not None:
        print('id 중복 train/test:', int(train[id_col].duplicated().sum()), int(test[id_col].duplicated().sum()),
              '| train 과 test 에 모두 있는 id 수:', len(set(train[id_col]) & set(test[id_col])) if id_col in train else '-')
    if sample is not None and test is not None and id_col:
        print('sample_submission 컬럼:', list(sample.columns), '| id 집합 == test:', set(sample[id_col]) == set(test[id_col]), '| 행 수', len(sample))
    if test is not None:                                          # 간단한 분포 이동 점검(평균 차이, 표준편차 단위)
        common = [c for c in num.columns if c in test.columns and c != id_col]
        if common:
            d = ((test[common].mean() - train[common].mean()) / (train[common].std() + 1e-9)).abs().sort_values(ascending=False)
            print('train/test 평균 차이 상위:', d.head(3).round(3).to_dict())
def target_structure(train, target, features=None, edge_share=0.05):
    """수치 타깃의 구조 단서: ① 최솟값·최댓값에 몰린 비율(검열·클리핑) ② 0 의 비율 ③ 기록 단위 ④ 선형 적합의 계수 합(가중 평균 구조)."""
    y = pd.to_numeric(train[target], errors='coerce').dropna().values.astype(float); n = len(y); hints = []
    lo, hi = y.min(), y.max(); p_lo, p_hi, p_zero = np.mean(y == lo), np.mean(y == hi), np.mean(y == 0)
    print(f'[{target}] n={n} 범위 {lo:g}~{hi:g} | 최솟값에 {p_lo:.1%}({int((y == lo).sum())}행), 최댓값에 {p_hi:.1%}({int((y == hi).sum())}행), 0 은 {p_zero:.1%}')
    unit = next((u for u in (1, 0.5, 0.25, 0.1, 0.05, 0.01) if np.allclose(y / u, np.round(y / u), atol=1e-6)), None)
    edge = 0.15 if (unit == 1 and len(np.unique(y)) <= 30) else edge_share      # 값 종류가 적은 정수 타깃은 끝값이 흔하므로 기준을 높인다
    if p_lo >= edge and (y == lo).sum() >= 3: hints.append(f'하한 {lo:g} 에 쏠림 -> 하한 검열 회귀(Tobit(lower={lo:g})) 또는 클리핑')
    if p_hi >= edge and (y == hi).sum() >= 3: hints.append(f'상한 {hi:g} 에 쏠림 -> 상한 검열 회귀(Tobit(upper={hi:g})) 또는 클리핑')
    if p_zero >= 0.2 and lo == 0 and p_lo < 1: hints.append('0 쏠림 -> 0 여부 분류 + 조건부 회귀도 후보')
    print(f'  기록 단위: {unit if unit is not None else "0.01 보다 잘거나 연속"} | 서로 다른 값 {len(np.unique(y))}개')
    if unit == 1 and len(np.unique(y)) <= 30: hints.append('정수 타깃 -> 제출 형식이 정수를 요구하는지 확인, 그룹 대표값은 중앙값')
    if features is not None and len(features) >= 2:
        X = train.loc[train[target].notna(), list(features)].apply(pd.to_numeric, errors='coerce')
        if not X.isna().any().any():
            Z = np.c_[np.ones(n), X.values.astype(float)]; b = np.linalg.lstsq(Z, y, rcond=None)[0]
            b0 = np.linalg.lstsq(X.values.astype(float), y, rcond=None)[0]; r2 = 1 - np.sum((y - Z @ b) ** 2) / np.sum((y - y.mean()) ** 2)
            print(f'  선형 적합: R2 {r2:.3f}, 절편 {b[0]:.3f}, 계수 {np.round(b[1:], 3).tolist()} | 절편 없이 적합한 계수 합 {b0.sum():.3f}')
            same_scale = X.values.min() >= lo - (hi - lo) and X.values.max() <= hi + (hi - lo)
            if abs(b0.sum() - 1) <= 0.05 and (b0 > 0).all() and same_scale: hints.append('계수 합이 1 에 가깝고 입력과 타깃의 척도가 같음 -> 가중 평균(WeightedMean)')
    print('  구조 단서:', ' / '.join(hints) if hints else '없음(일반 절차로 진행)')
    return hints

def box_check(X, y_pos):
    """이진 타깃이 '모든 기준을 통과하면 양성' 꼴인지 본다. 양성의 극값으로 만든 상자 안에 음성이 0명이면 규칙형(BoxRule) 단서."""
    X = pd.DataFrame(X); A = X.values.astype(float); y = np.asarray(y_pos).astype(bool)
    sign = np.where(A[y].mean(0) >= A[~y].mean(0), 1.0, -1.0); th = (A * sign)[y].min(0); inside = ((A * sign) >= th).all(1)
    drop = [int(((np.delete(A * sign, j, 1) >= np.delete(th, j)).all(1) & ~y).sum()) for j in range(A.shape[1])]
    print(f'양성 {int(y.sum())}명 / 음성 {int((~y).sum())}명 | 양성 극값 상자 안의 음성 {int((inside & ~y).sum())}명')
    for j, c in enumerate(X.columns):
        op = '>=' if sign[j] > 0 else '<='
        print(f'  {c} {op} {th[j] * sign[j]:g}  (이 기준을 빼면 상자 안 음성 {drop[j]}명)')
    ok = int((inside & ~y).sum()) == 0 and y.sum() >= 5
    print('  구조 단서:', '상자 안 음성 0명 -> 규칙형(BoxRule). 임계값이 양성 몇 명에 달렸는지와 test 양성 비율을 꼭 본다' if ok else '없음')
    return ok
