import sys, pickle, json, os, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import merge
VS = {"f640": ["full640"], "f1280": ["full1280"], "f640+960+1280": ["full640", "full960", "full1280"], "f960+1280+tile": ["full960", "full1280", "tile960"],
      "all7": ["full640", "flip640", "full960", "flip960", "full1280", "flip1280", "tile960"]}
p, out, cfg = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])     # cfg: {target: [viewset, nms, contain, vote, thr]}
V = pickle.load(open(f"_exp/{p}_views.pkl", "rb")); ss = pd.read_csv(f"{p}/data/sample_submission.csv"); tl = pd.read_csv(f"{p}/data/train_labels.csv")
sub = pd.DataFrame({"filename": ss.filename})
for tgt, (vs, nms, ct, vote, th) in cfg.items():
    sub[f"{tgt}_count"] = [int((merge(V[("test_images", f)], VS[vs], tgt, nms, ct, vote) >= th).sum()) for f in ss.filename]
    trp = np.array([int((merge(V[("train_images", f)], VS[vs], tgt, nms, ct, vote) >= th).sum()) for f in tl.filename]); y = tl[f"{tgt}_count"].values
    print(f"{tgt}: train MAE {np.abs(trp - y).mean():.3f} | 평균 수량 train 정답 {y.mean():.2f} / train 예측 {trp.mean():.2f} / test 예측 {sub[f'{tgt}_count'].mean():.2f} | test 분포 {dict(sub[f'{tgt}_count'].value_counts().sort_index())}")
sub = sub[list(ss.columns)]
assert sub.filename.tolist() == ss.filename.tolist() and not sub.isna().any().any() and (sub.iloc[:, 1:] >= 0).all().all()
os.makedirs(os.path.dirname(out), exist_ok=True); sub.to_csv(out, index=False); print("저장", out, sub.shape)
