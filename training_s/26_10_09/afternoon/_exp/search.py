"""관점 조합 x NMS x 포함도 x 투표 x 임계값 탐색. 정직한 추정은 LOO(29장으로 설정 선택 → 남은 1장 평가)."""
import sys, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import merge
p = sys.argv[1]; tl = pd.read_csv(f"{p}/data/train_labels.csv"); V = pickle.load(open(f"_exp/{p}_views.pkl", "rb"))
VS = {"f640": ["full640"], "f960": ["full960"], "f1280": ["full1280"], "f960+flip": ["full960", "flip960"], "f1280+flip": ["full1280", "flip1280"],
      "f640+960+1280": ["full640", "full960", "full1280"], "f1280+tile": ["full1280", "tile960"], "f960+1280+tile": ["full960", "full1280", "tile960"],
      "all7": ["full640", "flip640", "full960", "flip960", "full1280", "flip1280", "tile960"]}
TH = np.round(np.arange(0.05, 0.61, 0.025), 3)
for tgt in [c[:-6] for c in tl.columns if c.endswith("_count")]:
    y = tl[f"{tgt}_count"].values; names, P = [], []
    for vs, keys in VS.items():
        for nms in (0.4, 0.5, 0.6):
            for ct in (1.01, 0.8, 0.7):
                for vote in ((1, 2) if len(keys) >= 2 else (1,)):
                    sc = [merge(V[("train_images", f)], keys, tgt, nms, ct, vote) for f in tl.filename]
                    for th in TH:
                        names.append((vs, nms, ct, vote, float(th))); P.append([int((s >= th).sum()) for s in sc])
    P = np.array(P); E = np.abs(P - y)                  # [설정, 이미지]
    mae = E.mean(1); order = np.argsort(mae)
    print(f"=== {p} {tgt} | 설정 수 {len(names)} | 전부 평균값 예측 MAE {np.abs(y - np.round(y.mean())).mean():.3f}")
    for k in order[:6]: print(f"   in-sample MAE {mae[k]:.3f} 일치 {(P[k] == y).mean():.2f} 편향 {(P[k] - y).mean():+.2f} | {names[k]}")
    # 관점 조합별 최저
    for vs in VS:
        idx = [i for i, n in enumerate(names) if n[0] == vs]; b = idx[int(np.argmin(mae[idx]))]; print(f"   [{vs:16s}] 최저 {mae[b]:.3f} {names[b][1:]}")
    # LOO
    loo = []
    for i in range(len(y)):
        tr = np.delete(np.arange(len(y)), i); k = int(np.argmin(E[:, tr].mean(1))); loo.append(E[k, i])
    print(f"   LOO MAE(설정 전체에서 선택) {np.mean(loo):.3f} ± {np.std(loo) / np.sqrt(len(loo)):.3f}")
    # 넓은 고원 찾기: 설정 주변(임계값 ±0.05) 평균이 낮은 것
    pickle.dump((names, P, y), open(f"_exp/{p}_{tgt}_search.pkl", "wb"))
