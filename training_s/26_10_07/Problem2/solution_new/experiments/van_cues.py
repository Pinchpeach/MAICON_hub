# solution_new/experiments 폴더에서 실행 (경로는 상대 경로, OOF 점수는 v1_oof_score.npy)
import re, numpy as np, pandas as pd
OLD = r'v1_oof_score.npy'
tr = pd.read_csv(r'..\data\train.csv'); y = tr.label.values; s = np.load(OLD)
pred = (s >= np.quantile(s, 1 - y.mean())).astype(int); ok = pred == y; conf = np.abs(s - .5); rv = tr.review.values
cues = {
 '옹호/비난 대상 지칭(욕하지마·까지마·무뇌·뭐냐·니들·너희·평점테러·1점주)': r'(욕하지|까지\s*마|비난하지|평점\s*테러|1점\s*주|무뇌|뭐냐|니들|너희|꼴값|알바)',
 '명령형 금지(~하지마·마라·마세요·마셈·말아)': r'(하지\s*마|마라|마세요|마셈|말아|말자)',
 '비꼼 종결(떠나도 좋·잘 가시·굿\\.\\.·대단할순)': r'(떠나도\s*좋|잘\s*가시|굿\.\.|대단할순|ㅋ불쌍|ㅉㅉ)',
 '의문형 비난(?? 또는 ?로 끝나고 부정어 동반)': r'(\?\?|\?$)',
 '역접(참고용)': r'(하지만|그렇지만|그러나|그런데|근데|지만)',
 '인용/반박(왜 ~라고들|라고들|사람들 절반)': r'(라고들|라는데|라던데|사람들)',
}
print('기준 정확도 %.4f\n' % ok.mean())
for name, pat in cues.items():
    m = np.array([bool(re.search(pat, t)) for t in rv]); mc = m & (conf >= .2)
    for tag, mm in (('전체', m), ('확신도>=0.2', mc)):
        if mm.sum() == 0: continue
        acc = ok[mm].mean(); flip = 1 - acc
        print(f'{name[:44]:46s} {tag:8s} n={mm.sum():5d} | 현재 정확도 {acc:.3f} -> 뒤집으면 {flip:.3f} | 전체 정확도 변화 {((~ok[mm]).sum() - ok[mm].sum()) / len(tr) * 100:+.3f}%p')
print('\n홀수/짝수 행으로 나눠 안정성 확인 (옹호/비난 신호, 확신도>=0.2):')
m = np.array([bool(re.search(cues[list(cues)[0]], t)) for t in rv]) & (conf >= .2)
for nm, half in (('짝수 행', np.arange(len(tr)) % 2 == 0), ('홀수 행', np.arange(len(tr)) % 2 == 1)):
    mm = m & half; print(f'  {nm}: n={mm.sum()} 현재 정확도 {ok[mm].mean():.3f} 뒤집으면 {1 - ok[mm].mean():.3f}')
