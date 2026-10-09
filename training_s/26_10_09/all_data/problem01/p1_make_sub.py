"""최종 제출 생성: yolov8n 가중치(들) + 좌우 반전 TTA + 박스 융합(WBF, 클래스 무관 군집) + 임계값. 사용: python p1_make_sub.py <thr> <out.csv> <weights...>"""
import os, glob, sys, numpy as np, pandas as pd
from p1_ens import predict_raw, wbf, counts_from, NAMES
thr, out, W = float(sys.argv[1]), sys.argv[2], sys.argv[3:]
sample = pd.read_csv("data/sample_submission.csv")
by_name = {os.path.basename(p): p for p in glob.glob("data/test/images/*")}
assert set(sample.image_id) == set(by_name), "test 이미지와 sample ID 불일치"
paths = [by_name[i] for i in sample.image_id]
raws = [predict_raw(w, paths, flip=True) for w in W]
rows = []
for i, iid in enumerate(sample.image_id):
    cl = wbf([s for r in raws for s in r[i]], 2 * len(W), agnostic=True)
    rows.append([iid] + list(counts_from(cl, thr)))
sub = pd.DataFrame(rows, columns=["image_id"] + NAMES)
assert sub.image_id.tolist() == sample.image_id.tolist() and list(sub.columns) == list(sample.columns)
assert (sub[NAMES] >= 0).all().all() and not sub.isna().any().any() and len(sub) == 200
for c in NAMES: sub[c] = sub[c].astype(int)
sub.to_csv(out, index=False)
print(out, sub.shape, "합계", sub[NAMES].sum().to_dict(), "| 모두 0 이미지 %.3f" % (sub[NAMES].sum(1) == 0).mean())
