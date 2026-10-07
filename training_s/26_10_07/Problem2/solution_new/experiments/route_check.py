# solution_new/experiments 폴더에서 실행 (상대 경로). route_check.py 는 BiGRU OOF 파일(p2_rnn_oof.npy)이 필요함
import numpy as np, pandas as pd, os
OLDSP = r'C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-root-docker11\62eee907-2c50-4e3e-a181-3dcdb4a66b71\scratchpad\\'
tr = pd.read_csv(r'..\data\train.csv'); y = tr.label.values
s = np.load('v1_oof_score.npy'); rnn = np.load('p2_rnn_oof.npy')
print('파일 확인: v1 OOF', s.shape, '| BiGRU OOF', rnn.shape)
main = (s >= np.quantile(s, 1 - y.mean())).astype(int); nn_ = (rnn >= .5).astype(int)
rk = lambda v: pd.Series(v).rank().values / len(v)
print('전체 정확도: 1차(NB 앙상블) %.4f | BiGRU %.4f' % ((main == y).mean(), (nn_ == y).mean()))
gates = {'A. 긍정 예측 & s<0.6 (경계선 긍정)': (main == 1) & (s < .6), 'B. |s-0.5|<0.1 (경계선 전체)': np.abs(s - .5) < .1, 'C. |s-0.5|<0.2': np.abs(s - .5) < .2,
         'D. 긍정 예측 & s<0.7': (main == 1) & (s < .7)}
print('\n의심 구간(게이트)별 정확도: 1차 vs BiGRU(전체 데이터로 학습한 OOF) vs 평균, 그리고 게이트 안만 BiGRU 로 바꿨을 때 전체 정확도')
for name, g in gates.items():
    blend = ((rk(s) + rk(rnn)) / 2 >= .5)[g].astype(int)
    new_pred = main.copy(); new_pred[g] = nn_[g]
    print(f'  {name:32s} n={g.sum():5d} | 1차 {(main == y)[g].mean():.3f} | BiGRU {(nn_ == y)[g].mean():.3f} | 평균 {(blend == y[g]).mean():.3f} | 전체 정확도 {(new_pred == y).mean():.4f} ({((new_pred == y).mean() - (main == y).mean()) * 100:+.3f}%p)')
print('\nBiGRU 와 1차가 서로 다르게 예측하는 비율(게이트 안): ', {n[:1]: round(float((main != nn_)[g].mean()), 3) for n, g in gates.items()})
print('두 모델이 다를 때 BiGRU 가 맞는 비율(게이트 안):', {n[:1]: round(float((nn_ == y)[g & (main != nn_)].mean()), 3) for n, g in gates.items()})
