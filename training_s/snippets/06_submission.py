# 06_submission.py  ── 제출 파일 검증·저장. 제출 형식은 절대적: 컬럼명/순서/행 수/id 집합/결측·무한대/타입
# 사용:
#   sub = pd.DataFrame({'id': test['id'], 'price': pred})
#   check_submission(sub, test, 'id', ['id', 'price'], finite_cols=['price'], nonneg_cols=['price'])
#   save_submission(sub, 'submission.csv', versions_dir='versions', version_name='v1_hgb_rf.csv')   # 저장 후 다시 읽어 재검증
# 팁: index 열을 쓰지 않는다(index=False). 최종 노트북은 Restart & Run All 로 출력을 저장하고 마감(13.13).
import os, shutil
import numpy as np, pandas as pd

def check_submission(sub, test, id_col, columns, int_cols=(), finite_cols=(), nonneg_cols=(), allowed=None, sample=None):
    assert list(sub.columns) == list(columns), f'컬럼 불일치: {list(sub.columns)} != {list(columns)}'
    assert len(sub) == len(test), f'행 수 불일치: {len(sub)} != {len(test)}'
    assert sub[id_col].is_unique, 'id 중복'
    assert set(sub[id_col]) == set(test[id_col]), f'id 집합 불일치(누락 {len(set(test[id_col]) - set(sub[id_col]))}, 추가 {len(set(sub[id_col]) - set(test[id_col]))})'
    assert not sub.isna().any().any(), '결측(NaN) 존재'
    for c in int_cols: assert pd.api.types.is_integer_dtype(sub[c]), f'{c} 는 정수형이어야 함'
    for c in finite_cols: assert np.isfinite(sub[c]).all(), f'{c} 에 무한대/비수치 존재'
    for c in nonneg_cols: assert (sub[c] >= 0).all(), f'{c} 에 음수 존재'
    if allowed is not None:
        for c, vals in allowed.items(): assert set(sub[c].unique()) <= set(vals), f'{c} 에 허용되지 않은 값'
    if sample is not None:
        assert list(sub.columns) == list(sample.columns) and set(sub[id_col]) == set(sample[id_col]), 'sample_submission 과 불일치'
    print(f'제출 형식 OK: {len(sub)}행, 컬럼 {list(sub.columns)}')

def save_submission(sub, path='submission.csv', versions_dir=None, version_name=None):
    sub.to_csv(path, index=False)                                # index 열 금지
    chk = pd.read_csv(path)
    assert list(chk.columns) == list(sub.columns) and len(chk) == len(sub), '저장 후 재검증 실패'
    if versions_dir and version_name:
        os.makedirs(versions_dir, exist_ok=True); shutil.copy(path, os.path.join(versions_dir, version_name))
    print('저장:', path, '| 버전 보존:', os.path.join(versions_dir, version_name) if versions_dir and version_name else '없음')
    return chk
