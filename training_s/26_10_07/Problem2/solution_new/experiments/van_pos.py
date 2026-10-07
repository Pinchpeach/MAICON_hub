# solution_new/experiments 폴더에서 실행 (경로는 상대 경로)
import re, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, average_precision_score
OLD = r'v1_oof_score.npy'
N = r'..\\'
tr = pd.read_csv(N + 'data/train.csv'); te = pd.read_csv(N + 'data/test.csv'); base = pd.read_csv(N + 'submission_base.csv')
y = tr.label.values; s = np.load(OLD); pred = (s >= np.quantile(s, 1 - y.mean())).astype(int)
P = np.where(pred == 1)[0]; rv = tr.review.values[P]; yP = y[P]; sP = s[P]
wrong = (yP == 0).astype(int)                     # 1차가 긍정이라 했지만 실제 부정 = 틀린 긍정(반어/거짓 긍정)
van = (wrong == 1) & (sP - .5 >= .25)             # 그 중 확신하는데 틀린 것(train_van 의 긍정 예측 부분)
print(f'1차 긍정 예측 {len(P)}개 | 틀린 긍정 {wrong.sum()}개 ({wrong.mean() * 100:.1f}%) | 확신 오류(van) {van.sum()}개')
CUES = [r'(하지만|그렇지만|그러나|그런데|근데|지만)', r'(욕하지|마라|마세요|마셈|무뇌|뭐냐|니들|너희)', r'(떠나도\s*좋|잘\s*가시|굿\.\.|대단할순|ㅋㅋ)', r'(\?\?|\?$)', r'(\.\.|…)', r'(전부|뿐|밖에|만\s*(크|좋|멋|예쁘))', r'(1점|0점|별로|아깝|돈)', r'(ㅡㅡ|ㅠㅠ|;;)']
def cue(texts): return csr_matrix(np.array([[1.0 if re.search(c, t) else 0.0 for c in CUES] for t in texts]))
def sfeat(sc): c = sc - .5; return csr_matrix(np.column_stack([c * 4, (c > .25).astype(float), (c > .35).astype(float)]))
def vecs(): return [TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True, min_df=2), TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, token_pattern=r'\S+', min_df=2)]
def build(texts_fit, sc_fit, texts_app, sc_app, use_s, use_cue):
    vs = vecs(); A = [v.fit_transform(texts_fit) for v in vs]; B = [v.transform(texts_app) for v in vs]
    if use_s: A.append(sfeat(sc_fit)); B.append(sfeat(sc_app))
    if use_cue: A.append(cue(texts_fit)); B.append(cue(texts_app))
    return hstack(A).tocsr(), hstack(B).tocsr()
def cv_oof(use_s, use_cue, C=0.5):
    o = np.zeros(len(P))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(rv, wrong):
        Xa, Xb = build(rv[a], sP[a], rv[b], sP[b], use_s, use_cue)
        o[b] = LogisticRegression(C=C, class_weight='balanced', max_iter=3000).fit(Xa, wrong[a]).predict_proba(Xb)[:, 1]
    return o
base_rate = wrong.mean(); R = {}
print(f'\n[1차 긍정 예측 {len(P)}개 안에서 "틀린 긍정"을 가려내는 2차 검증기, 5-Fold CV, 무작위 기준 AP={base_rate:.3f}]')
for name, kw in [('텍스트만', dict(use_s=False, use_cue=False)), ('텍스트+반어 단서', dict(use_s=False, use_cue=True)), ('텍스트+1차 점수', dict(use_s=True, use_cue=False)), ('텍스트+점수+단서', dict(use_s=True, use_cue=True))]:
    o = cv_oof(**kw); R[name] = o
    print(f'  {name:16s} AUC(틀린 긍정) {roc_auc_score(wrong, o):.3f} | AP {average_precision_score(wrong, o):.3f} | AUC(확신 오류) {roc_auc_score(van, o):.3f}')
print('  [참고] 1차 점수만(낮을수록 틀린 긍정일 가능성): AUC %.3f' % roc_auc_score(wrong, -sP))
best = max(R, key=lambda k: average_precision_score(wrong, R[k])); o = R[best]; order = np.argsort(-o)
print(f'\n[{best}] 상위 k개 정밀도(= 그 중 실제 틀린 긍정 비율, 기준 {base_rate:.3f}) / 뒤집었을 때 정확도 변화')
print('  ' + ' | '.join(f'k={k}: {wrong[order[:k]].mean():.2f} ({wrong[order[:k]].mean() / base_rate:.1f}배), {((wrong[order[:k]].sum() - (1 - wrong[order[:k]]).sum()) / len(tr)) * 100:+.3f}%p' for k in (20, 50, 100, 200, 500, 1000)))
pd.set_option('display.max_colwidth', 105); pd.set_option('display.width', 250)
print(f'\n[{best}] 점수 상위 25개 (y=실제 라벨, 0이면 틀린 긍정)'); [print(f'  p={o[i]:.2f} s={sP[i]:.2f} y={yP[i]} | {rv[i][:100]}') for i in order[:25]]
np.save(N + r'experiments\van_pos_oof.npy', o)
# ---- 테스트: 현재 제출의 긍정 문장에 2차 검증기 점수 (뒤집지 않음) ----
class NBLR(BaseEstimator, ClassifierMixin):
    def __init__(self, C=8, alpha=1.0): self.C = C; self.alpha = alpha
    def _vec(self): return [CountVectorizer(analyzer='char_wb', ngram_range=(1, 4), binary=True, min_df=2), CountVectorizer(analyzer='word', ngram_range=(1, 2), binary=True, token_pattern=r'\S+', min_df=2)]
    def fit(self, T, y):
        self.v_ = [v.fit(T) for v in self._vec()]; M = hstack([v.transform(T) for v in self.v_]).tocsr()
        p = self.alpha + M[y == 1].sum(0); q = self.alpha + M[y == 0].sum(0); self.r_ = np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()
        self.lr_ = LogisticRegression(C=self.C, max_iter=3000).fit(M.multiply(self.r_).tocsr(), y); self.classes_ = self.lr_.classes_; return self
    def predict_proba(self, T): return self.lr_.predict_proba(hstack([v.transform(T) for v in self.v_]).tocsr().multiply(self.r_).tocsr())
trio = [make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True, min_df=2), ComplementNB(alpha=0.5)),
        make_pipeline(CountVectorizer(analyzer='char_wb', ngram_range=(1, 4), binary=True, min_df=2), MultinomialNB(alpha=0.5)), NBLR(C=8)]
rank = lambda p: pd.Series(p).rank().values / len(p)
s_te = np.mean([rank(m.fit(tr.review.values, y).predict_proba(te.review.values)[:, 1]) for m in trio], axis=0)
pos = np.where(base.label.values == 1)[0]
Xa, Xb = build(rv, sP, te.review.values[pos], s_te[pos], True, True)
p_te = LogisticRegression(C=0.5, class_weight='balanced', max_iter=3000).fit(Xa, wrong).predict_proba(Xb)[:, 1] if best == '텍스트+점수+단서' else None
if p_te is None:
    use_s, use_cue = ('점수' in best), ('단서' in best)
    Xa, Xb = build(rv, sP, te.review.values[pos], s_te[pos], use_s, use_cue)
    p_te = LogisticRegression(C=0.5, class_weight='balanced', max_iter=3000).fit(Xa, wrong).predict_proba(Xb)[:, 1]
out = pd.DataFrame({'id': te.id.values[pos], 'base_label': 1, 'v1_score_test': s_te[pos].round(4), 'van_verifier_p': p_te.round(4), 'review': te.review.values[pos]}).sort_values('van_verifier_p', ascending=False)
out.to_csv(N + 'van_verifier_test_scores.csv', index=False)
print(f'\n[테스트] 현재 제출의 긍정 {len(pos)}개에 검증기 점수 저장 (제출 값은 변경하지 않음). 사용한 검증기: {best}')
thr = np.sort(o)[::-1][99]; print(f'  train OOF 상위 100개 점수 기준값 {thr:.2f} 이상인 테스트 긍정: {(p_te >= thr).sum()}개 | 점수 분위(50/90/99%): {np.percentile(p_te, [50, 90, 99]).round(2)}')
print('  테스트 점수 상위 12개:'); [print(f'    p={r.van_verifier_p:.2f} | {r.review[:95]}') for r in out.head(12).itertuples()]
