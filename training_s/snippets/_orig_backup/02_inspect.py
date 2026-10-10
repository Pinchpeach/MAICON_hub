# 02_inspect.py  (T+0~5, 약 2분)  ── 데이터 점검: 결측·표기 변형·0값·중복·id·타깃·분포 이동
# 사용: inspect_data(train, test, target='price', id_col='id', sample=sample_submission)
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
