"""사전학습 yolov8s(COCO)만으로 만든 기준선과 그 안정성. 정답이 없으므로 구조(종류 결정, 개수의 임계값 민감도)만 본다."""
import os, pickle, collections, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
AIR, BIRD = 4, 14; R = pickle.load(open("_p2_coco_pass.pkl", "rb")); tf = sorted(R); os.makedirs("versions", exist_ok=True)

def decide(f, thr=0.25, nms=0.6):
    c, s, b = R[f]; ta = s[c == AIR].max() if (c == AIR).any() else 0.0; tb = s[c == BIRD].max() if (c == BIRD).any() else 0.0
    if max(ta, tb) < thr: return "drone", 1, ta, tb
    k = AIR if ta >= tb else BIRD; m = (c == k) & (s >= thr); bx = b[m]; sc = s[m]
    keep = []                                                    # 같은 종류끼리 IoU 중복 제거
    for i in np.argsort(-sc):
        ok = True
        for j in keep:
            iw = max(0, min(bx[i, 2], bx[j, 2]) - max(bx[i, 0], bx[j, 0])); ih = max(0, min(bx[i, 3], bx[j, 3]) - max(bx[i, 1], bx[j, 1])); inter = iw * ih
            u = (bx[i, 2] - bx[i, 0]) * (bx[i, 3] - bx[i, 1]) + (bx[j, 2] - bx[j, 0]) * (bx[j, 3] - bx[j, 1]) - inter
            if inter / (u + 1e-12) >= nms: ok = False; break
        if ok: keep.append(i)
    return ("airplane" if k == AIR else "bird"), len(keep), ta, tb

rows = []
for f in tf:
    kind, n, ta, tb = decide(f); rows.append(dict(filename=f, airplane=n if kind == "airplane" else 0, bird=n if kind == "bird" else 0, drone=n if kind == "drone" else 0, kind=kind, top_air=ta, top_bird=tb))
S = pd.DataFrame(rows); S[["filename", "airplane", "bird", "drone"]].to_csv("versions/v1_pretrained_only_baseline.csv", index=False)
print("== 기준선 (사전학습 yolov8s, conf 0.25, 비행기·새 없음 = 드론 1대) ==")
print("종류:", dict(S.kind.value_counts()), "| 수량 합: 비행기 %d, 새 %d, 드론 %d" % (S.airplane.sum(), S.bird.sum(), S.drone.sum()))
print("\n== 임계값 민감도: 이미지의 종류·수량이 임계값에 따라 바뀌는가 ==")
T = (0.1, 0.15, 0.25, 0.35, 0.5); res = {t: [decide(f, t)[:2] for f in tf] for t in T}
for t in T: print("  conf %.2f: 종류 %s" % (t, dict(collections.Counter(k for k, n in res[t]))))
base = res[0.25]; chg_kind = [f for f, a, b in zip(tf, res[0.15], res[0.35]) if a[0] != b[0]]; chg_n = [f for f, a, b in zip(tf, res[0.15], res[0.35]) if a[0] == b[0] and a[1] != b[1]]
print("conf 0.15 와 0.35 사이에서 종류가 바뀌는 이미지 %d장, 수량만 바뀌는 이미지 %d장 (전체 300장 중)" % (len(chg_kind), len(chg_n)))
pd.DataFrame({"f": chg_kind + chg_n, "type": ["kind"] * len(chg_kind) + ["count"] * len(chg_n)}).to_csv("_p2_unstable.csv", index=False)
print("  종류가 바뀌는 예:", chg_kind[:10]); print("  수량이 바뀌는 예:", chg_n[:10])
print("\n== 종류 결정 여유: 1위 종류 점수와 2위 종류 점수 ==")
S["margin"] = (S.top_air - S.top_bird).abs(); S["top"] = S[["top_air", "top_bird"]].max(axis=1)
amb = S[(S.top >= 0.25) & (S.top_air >= 0.1) & (S.top_bird >= 0.1)]; print("두 종류 모두 점수 0.1 이상인 이미지 %d장:" % len(amb), amb.filename.tolist()[:12])
low = S[(S.kind != "drone") & (S.top < 0.5)]; print("비행기·새로 정했지만 점수가 0.5 미만인 이미지 %d장 (판정이 약함)" % len(low))
dr = S[S.kind == "drone"]; print("드론으로 정한 %d장 중 비행기·새 점수가 0.05~0.25 사이로 걸쳤던 이미지 %d장" % (len(dr), ((dr.top_air > 0.05) | (dr.top_bird > 0.05)).sum()))
cnt = S[S.kind != "drone"]; cnt = cnt.assign(n=cnt.airplane + cnt.bird); print("수량 분포(비행기·새): %s | 수량 2 이상 %d장" % (cnt.n.value_counts().sort_index().to_dict(), (cnt.n >= 2).sum()))
S.to_csv("_p2_baseline_detail.csv", index=False)
