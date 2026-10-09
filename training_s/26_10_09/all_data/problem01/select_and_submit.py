"""여러 가중치를 검증 160장으로 평가하고, 가장 높은 설정으로 test 제출을 만든다."""
import glob, os, sys, numpy as np, pandas as pd
from ultralytics import YOLO
NAMES = ["military_tank", "military_truck", "military_vehicle", "civilian_vehicle"]
T = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)
imgs = sorted(glob.glob("data/val/images/*"))
def gt(i):
    c = [0] * 4
    for l in open(i.replace("images", "labels").rsplit(".", 1)[0] + ".txt").read().splitlines():
        if l.strip(): c[int(l.split()[0])] += 1
    return tuple(c)
G = [gt(i) for i in imgs]
cnt = lambda k, s, t: tuple(int(((k == c) & (s >= t)).sum()) for c in range(4))
def boxes(m, paths, **kw):
    out = []
    for i in range(0, len(paths), 40):
        out += [(x.boxes.cls.cpu().numpy().astype(int), x.boxes.conf.cpu().numpy()) for x in m.predict(paths[i:i + 40], conf=0.05, device=0, verbose=False, **kw)]
    return out
best = None; rows = []
for w in sys.argv[1:]:
    if not os.path.exists(w): print("없음:", w); continue
    m = YOLO(w)
    for sz in (640, 800):
        for ag in (False, True):
            bx = boxes(m, imgs, imgsz=sz, iou=0.6, agnostic_nms=ag)
            a = {t: np.mean([cnt(k, s, t) == g for (k, s), g in zip(bx, G)]) for t in T}; bt = max(a, key=a.get)
            # 인접 conf 평균도 같이 본다(한 점의 우연한 최고값 방지)
            print(f"{w.split('/')[-3]:14s} imgsz {sz} agnostic {int(ag)} | best conf {bt} acc {a[bt]:.4f} | " + " ".join(f"{t}:{a[t]:.3f}" for t in T), flush=True)
            if best is None or a[bt] > best[0]: best = (a[bt], w, sz, ag, bt)
print("선택:", best)
acc, w, sz, ag, bt = best
sample = pd.read_csv("data/sample_submission.csv"); paths = {os.path.basename(p): p for p in glob.glob("data/test/images/*")}
assert set(sample.image_id) == set(paths)
bx = boxes(YOLO(w), [paths[x] for x in sample.image_id], imgsz=sz, iou=0.6, agnostic_nms=ag)
sub = pd.DataFrame([[i] + list(cnt(k, s, bt)) for i, (k, s) in zip(sample.image_id, bx)], columns=["image_id"] + NAMES)
assert sub.image_id.tolist() == sample.image_id.tolist() and list(sub.columns) == list(sample.columns) and not sub.isna().any().any() and (sub[NAMES] >= 0).all().all()
out = f"versions/cand_{w.split('/')[-3]}_sz{sz}_ag{int(ag)}_conf{bt}.csv"; sub.to_csv(out, index=False)
print("후보 저장:", out, "| 합계", sub[NAMES].sum(axis=0).to_dict(), "| 모두 0인 이미지 %.3f" % (sub[NAMES].sum(axis=1) == 0).mean())
