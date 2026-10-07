# solution_new/experiments 폴더에서 실행 (경로는 상대 경로, OOF 점수는 v1_oof_score.npy)
import numpy as np, pandas as pd
OLD = r'v1_oof_score.npy'
tr = pd.read_csv(r'..\data\train.csv')
s = np.load(OLD); y = tr.label.values
pred = (s >= np.quantile(s, 1 - y.mean())).astype(int); wrong = pred != y; conf = np.abs(s - 0.5)
print('train', len(tr), '| v1 OOF 정확도 %.4f | 오류 %d개' % ((~wrong).mean(), wrong.sum()))
print('확신도(|s-0.5|) 구간별: 행 수 / 오류 수 / 오류율')
for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5)]:
    m = (conf >= lo) & (conf < hi); print(f'  [{lo:.1f},{hi:.1f}) n={m.sum():5d} wrong={wrong[m].sum():4d} err={wrong[m].mean():.3f}')
print('\n후보 기준별 개수 (확신도 >= t 이면서 오류):')
for t in (.15, .2, .25, .3, .35, .4):
    m = wrong & (conf >= t); print(f'  t={t:.2f}: {m.sum():4d}개 ({m.mean() * 100:.2f}%) | 라벨 1(긍정)인데 부정 예측 {int((m & (y == 1)).sum())} / 라벨 0인데 긍정 예측 {int((m & (y == 0)).sum())}')
pd.set_option('display.max_colwidth', 120); pd.set_option('display.width', 250)
m = wrong & (conf >= .25); idx = np.where(m)[0]; rng = np.random.default_rng(0); pick = rng.choice(idx, 45, replace=False); pick.sort()
print(f'\n[검수용 표본] 확신도>=0.25 오류 {m.sum()}개 중 무작위 45개 (label=정답, pred=모델)')
for i in pick: print(f'{i:5d} y={y[i]} pred={pred[i]} s={s[i]:.2f} | {tr.review[i][:115]}')
