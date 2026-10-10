"""2차 검증: 1차 규칙에서 탈락한 저확신 후보를 후보 중심으로 크게 확대(zoom)해 yolov8n 으로 다시 검출한다.
출력: _exp/<problem>_<target>_stage2.pkl  (후보별 특징: 1차 conf, 관점 수, 크기, 확대 재검출 conf 등)
사용: python _exp/stage2.py <problem> <target>"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "_exp")
from lib import _iou_inter, GROUPS
from ultralytics import YOLO
from PIL import Image

p, tgt = sys.argv[1], sys.argv[2]
CFG = {("problem02", "person"): (["full960", "full1280", "tile960"], 0.5, 0.7, 2, 0.15),
       ("problem04", "person"): (["full640", "flip640", "full960", "flip960", "full1280", "flip1280"], 0.5, 0.8, 1, 0.15),
       ("problem04", "vehicle"): (["full640", "full960", "full1280"], 0.5, 0.7, 2, 0.175)}
KEYS, NMS, CT, VOTE, THR = CFG[(p, tgt)]; G = GROUPS[tgt]
V = pickle.load(open(f"_exp/{p}_views.pkl", "rb")); m = YOLO(f"{p}/yolov8n.pt")

def clusters(views):
    B, S, Vw = [], [], []
    for vi, k in enumerate(KEYS):
        b, s, c = views[k]; mk = np.isin(c, G); B.append(b[mk]); S.append(s[mk]); Vw.append(np.full(int(mk.sum()), vi))
    B, S, Vw = map(np.concatenate, (B, S, Vw)); reps, mem = [], []
    for i in np.argsort(-S):
        if reps:
            io, ct = _iou_inter(B[i], B[reps]); j = np.where((io >= NMS) | (ct >= CT))[0]
            if len(j): mem[j[0]].append(i); continue
        reps.append(i); mem.append([i])
    out = []
    for r, mm in zip(reps, mem):
        best = {}
        for i in mm: best[Vw[i]] = max(best.get(Vw[i], 0), S[i])
        out.append((B[r], float(S[r]), sum(best.values()) / len(KEYS), len(best)))
    return out

def zoom_conf(arr, box, scale, out_sz=640):
    """후보 박스 중심으로 (박스 긴 변 x scale, 최소 64px) 정사각 영역을 잘라 out_sz 로 키워 재검출. 후보와 겹치는 대상 클래스 박스의 최고 conf."""
    H, W = arr.shape[:2]; x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    side = max(64.0, scale * max(x2 - x1, y2 - y1)); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ax1, ay1 = int(max(0, min(W - side, cx - side / 2))), int(max(0, min(H - side, cy - side / 2))); ax2, ay2 = int(min(W, ax1 + side)), int(min(H, ay1 + side))
    crop = arr[ay1:ay2, ax1:ax2].copy(); best = 0.0
    for flip in (False, True):
        r = m.predict(crop[:, ::-1].copy() if flip else crop, imgsz=out_sz, conf=0.03, iou=0.7, device=0, verbose=False, max_det=100)[0]
        b = r.boxes.xyxyn.cpu().numpy().copy(); s = r.boxes.conf.cpu().numpy(); c = r.boxes.cls.cpu().numpy().astype(int)
        if flip: b[:, [0, 2]] = 1 - b[:, [2, 0]]
        mk = np.isin(c, G)
        if not mk.any(): continue
        b = b[mk]; s = s[mk]; cw, ch = ax2 - ax1, ay2 - ay1
        bb = np.stack([(ax1 + b[:, 0] * cw) / W, (ay1 + b[:, 1] * ch) / H, (ax1 + b[:, 2] * cw) / W, (ay1 + b[:, 3] * ch) / H], 1)
        io, ct = _iou_inter(np.asarray(box), bb); ok = (io >= 0.3) | (ct >= 0.6)
        if ok.any(): best = max(best, float(s[ok].max()))
    return best

rows = []
for (split, f), views in V.items():
    arr = np.asarray(Image.open(f"{p}/data/{split}/{f}").convert("RGB"))[:, :, ::-1]; H, W = arr.shape[:2]
    cl = clusters(views); accepted = [c for c in cl if c[3] >= VOTE and c[2] >= THR]
    for box, mx, sc, nv in cl:
        counted = nv >= VOTE and sc >= THR; row = dict(split=split, f=f, counted=counted, maxconf=mx, score=sc, nviews=nv, h_px=(box[3] - box[1]) * H, w_px=(box[2] - box[0]) * W,
                                                    cy=(box[1] + box[3]) / 2, cx=(box[0] + box[2]) / 2, box=box)
        if not counted:
            row["z3"] = zoom_conf(arr, box, 3.0); row["z6"] = zoom_conf(arr, box, 6.0)
            # 이미 세어진 객체와 겹침(중복 후보인지)
            if accepted:
                io, ct = _iou_inter(np.asarray(box), np.array([a[0] for a in accepted])); row["ov_acc"] = float(max(io.max(), ct.max()))
            else: row["ov_acc"] = 0.0
        rows.append(row)
D = pd.DataFrame(rows); pickle.dump(D, open(f"_exp/{p}_{tgt}_stage2.pkl", "wb"))
c = D[~D.counted]; print(p, tgt, "전체 군집", len(D), "| 1차 탈락 후보", len(c), "| z3>=0.25:", int((c.z3 >= 0.25).sum()), "| z6>=0.25:", int((c.z6 >= 0.25).sum()))
