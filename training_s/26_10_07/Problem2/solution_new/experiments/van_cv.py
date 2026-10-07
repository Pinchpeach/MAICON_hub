# solution_new/experiments 폴더에서 실행 (경로는 상대 경로, OOF 점수는 v1_oof_score.npy)
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
OLD = r'v1_oof_score.npy'
N = r'..\\'
SP = r'C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-root-docker11\7ec5aa84-89af-4967-a0fe-dd8714961a0f\scratchpad\\'
tr = pd.read_csv(N + 'data/train.csv'); y = tr.label.values; s = np.load(OLD)
pred = (s >= np.quantile(s, 1 - y.mean())).astype(int); wrong = pred != y; conf = np.abs(s - 0.5)
van = (wrong & (conf >= 0.25)).astype(int)
# ---- 1. train_van.csv (반어/모순 후보) 와 수동 검수 표본 저장 ----
out = tr.copy(); out['oof_score'] = s.round(4); out['model_pred'] = pred; out['van'] = van
out[van == 1].to_csv(N + 'train_van.csv', index=False)
cat = {}
for k in (120, 714, 1476, 6843, 8962, 9245, 10246, 10420, 11218, 11727, 11772, 12689, 12923, 15165, 15645, 16960, 17386, 18366): cat[k] = 'irony_defense(반어·옹호·표면과 반대)'
for k in (141, 636, 1727, 1957, 2798, 3874, 10664, 13978, 14767, 15783, 16297, 17403, 18055, 19148): cat[k] = 'label_noise(라벨 노이즈·중립)'
for k in (96, 384, 536, 671, 3490, 5176, 5750, 6376, 8331, 13145, 13382, 15987, 19431): cat[k] = 'contrast(역접·복합)'
aud = tr.loc[list(cat)].copy(); aud['row'] = list(cat); aud['manual_category'] = [cat[k] for k in cat]; aud['oof_score'] = s[list(cat)].round(3)
aud[['row', 'id', 'review', 'label', 'oof_score', 'manual_category']].to_csv(N + 'train_van_audit.csv', index=False)
print('train_van.csv', int(van.sum()), '행 | 검수 표본', len(aud), '개:', aud.manual_category.value_counts().to_dict())
# ---- 2. 반어 탐지 모델 (반어=1, 아니면=0): 5-Fold CV ----
def feats(): return [TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True, min_df=2), TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, token_pattern=r'\S+', min_df=2)]
def dense(sc): c = np.abs(sc - 0.5); return csr_matrix(np.column_stack([c * 4, (sc > .5).astype(float), (c > .25).astype(float), (c > .35).astype(float)]))
def fit_pred(a, b, use_s=True, C=1.0):
    vs = feats(); Xa = hstack([v.fit_transform(tr.review.values[a]) for v in vs] + ([dense(s[a])] if use_s else [])).tocsr()
    Xb = hstack([v.transform(tr.review.values[b]) for v in vs] + ([dense(s[b])] if use_s else [])).tocsr()
    return LogisticRegression(C=C, class_weight='balanced', max_iter=3000).fit(Xa, van[a]).predict_proba(Xb)[:, 1]
res = {}
for name, kw in [('텍스트+점수 C=1', dict(use_s=True, C=1.0)), ('텍스트+점수 C=0.2', dict(use_s=True, C=0.2)), ('텍스트만 C=1', dict(use_s=False, C=1.0))]:
    p = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(tr, van): p[b] = fit_pred(a, b, **kw)
    res[name] = p
    print(f'{name:18s} 반어 탐지 AUC {roc_auc_score(van, p):.3f}  (van 비율 {van.mean() * 100:.2f}%, 평균정밀도 기준 무작위={van.mean():.3f})')
# ---- 3. 뒤집기 시뮬레이션: 탐지 점수 상위 k 개를 뒤집었을 때 정확도 ----
base_acc = float((~wrong).mean()); print('\n기준(v1 OOF) 정확도 %.4f (오류 %d개)' % (base_acc, wrong.sum()))
for name, p in res.items():
    order = np.argsort(-p); print(f'[{name}] 상위 k 개 뒤집기: k, 뒤집은 것 중 실제 오류 비율(정밀도), 정확도 변화')
    row = []
    for k in (10, 25, 50, 100, 200, 400, 800):
        idx = order[:k]; prec = wrong[idx].mean(); gain = (wrong[idx].sum() - (~wrong[idx]).sum()) / len(tr)
        row.append(f'k={k}: 정밀도 {prec:.2f}, 변화 {gain * 100:+.3f}%p')
    print('   ' + ' | '.join(row))
np.save(SP + 'van_oof_p.npy', res['텍스트+점수 C=1'])
