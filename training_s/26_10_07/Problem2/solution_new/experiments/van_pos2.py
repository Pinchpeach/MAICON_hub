# solution_new/experiments 폴더에서 실행 (경로는 상대 경로)
import re, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open(r'C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-root-docker11\7ec5aa84-89af-4967-a0fe-dd8714961a0f\scratchpad\van_pos.py', encoding='utf-8').read()
exec(src.split("base_rate = wrong.mean(); R = {}")[0])          # 데이터/특징 정의만 재사용
o_txt = cv_oof(use_s=False, use_cue=True)                         # 텍스트+단서 (점수 미사용)
o_all = np.load(N + r'experiments\van_pos_oof.npy')               # 텍스트+점수+단서
rk = lambda v: pd.Series(v).rank().values / len(v)
print('1) 1차 점수 구간별 "틀린 긍정" 비율, 그리고 구간 안에서 텍스트 검증기(점수 미사용)의 AUC')
for lo, hi in [(.5, .55), (.55, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
    m = (sP >= lo) & (sP < hi)
    auc = roc_auc_score(wrong[m], o_txt[m]) if 0 < wrong[m].sum() < m.sum() else float('nan')
    print(f'   s in [{lo:.2f},{hi:.2f}): n={m.sum():5d} 틀린 긍정 비율 {wrong[m].mean():.3f} | 텍스트 검증기 AUC {auc:.3f}')
print('\n2) 전체 긍정 예측에서 "틀린 긍정" 가려내기 AUC / AP')
cmp = {'1차 점수만 (낮을수록 의심)': -sP, '텍스트 검증기만(점수 미사용)': o_txt, '텍스트+점수 검증기(LR)': o_all,
       '순위합: 1차 점수 + 텍스트 검증기': rk(-sP) + 0.5 * rk(o_txt), '순위합 가중 1:1': rk(-sP) + rk(o_txt), '순위합 가중 1:0.25': rk(-sP) + 0.25 * rk(o_txt)}
for k, v in cmp.items(): print(f'   {k:32s} AUC {roc_auc_score(wrong, v):.3f} | AP {average_precision_score(wrong, v):.3f} | 상위100 정밀도 {wrong[np.argsort(-v)[:100]].mean():.2f} | 상위500 {wrong[np.argsort(-v)[:500]].mean():.2f}')
print('\n3) 경계선(s<0.6) 긍정 예측만 따로 봤을 때: 뒤집었을 때 정확도 변화')
b = np.where(sP < .6)[0]; print(f'   s<0.6 긍정 예측 {len(b)}개 중 틀린 긍정 {wrong[b].sum()}개 ({wrong[b].mean():.3f})  -> 전부 뒤집으면 {(wrong[b].sum() - (1 - wrong[b]).sum()) / len(tr) * 100:+.3f}%p')
