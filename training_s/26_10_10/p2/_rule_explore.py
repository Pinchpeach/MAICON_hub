import pickle, collections, numpy as np, pandas as pd
D = pickle.load(open("_final_preds.pkl", "rb")); PD, PB = D["P_drone"], D["P_base"]; AIR, BIRD = 4, 14; fs = sorted(PD)
rows = []
for f in fs:
    c, s, b = PB[f]; sd = float(PD[f][1].max()) if len(PD[f][1]) else 0.0
    rows.append(dict(f=f, sd=sd, sa=float(s[c == AIR].max()) if (c == AIR).any() else 0.0, sb=float(s[c == BIRD].max()) if (c == BIRD).any() else 0.0))
T = pd.DataFrame(rows); T["sp"] = T[["sa", "sb"]].max(axis=1)
print("== 드론 모델 최고 점수(sd) 분포 ==", np.histogram(T.sd, bins=[0, .1, .2, .3, .35, .5, .7, .9, 1.01])[0].tolist(), "(구간: 0-.1 .1-.2 .2-.3 .3-.35 .35-.5 .5-.7 .7-.9 .9-1)")
print("== 기본 모델 비행기·새 최고 점수(sp) 분포 ==", np.histogram(T.sp, bins=[0, .1, .2, .3, .35, .5, .7, .9, 1.01])[0].tolist())
ct = pd.crosstab(pd.cut(T.sd, [-.01, .1, .2, .35, .5, .7, 1.01], labels=["<.1", ".1-.2", ".2-.35", ".35-.5", ".5-.7", ">.7"]), pd.cut(T.sp, [-.01, .1, .25, .5, .7, 1.01], labels=["<.1", ".1-.25", ".25-.5", ".5-.7", ">.7"]))
print("\n== 2차원 분포: 행 = 드론 모델 점수, 열 = 기본 모델(비행기·새) 점수 ==\n" + ct.to_string())
conf = T[(T.sd >= 0.35) & (T.sp >= 0.25)].sort_values("sp", ascending=False)
print("\n== 두 모델이 모두 확신하는 충돌 이미지 (sd >= 0.35 이고 sp >= 0.25): %d장 ==" % len(conf))
print(conf.round(2).to_string(index=False))
only_d = T[(T.sd >= 0.35) & (T.sp < 0.25)]; only_p = T[(T.sd < 0.35) & (T.sp >= 0.25)]; neither = T[(T.sd < 0.35) & (T.sp < 0.25)]
print("\n드론만 확신 %d장 | 기본만 확신 %d장 | 둘 다 확신 %d장 | 둘 다 아님 %d장" % (len(only_d), len(only_p), len(conf), len(neither)))
print("둘 다 아님:", neither.round(2).to_string(index=False) if len(neither) else "없음")
T.to_csv("_rule_scores.csv", index=False)
