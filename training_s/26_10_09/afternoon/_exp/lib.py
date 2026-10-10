"""수량 예측 공용 함수: 전체 이미지/좌우 반전/타일 추론으로 원시 박스를 모으고, 묶음별 NMS·포함도 억제 후 개수를 센다."""
import numpy as np
from PIL import Image

GROUPS = {"person": [0], "vehicle": [2, 5, 7]}   # COCO class id

def _collect(res, x0=0.0, y0=0.0, sx=1.0, sy=1.0, flip=False):
    """예측 결과를 전체 이미지 기준 정규화 xyxy 로 변환."""
    b = res.boxes.xyxyn.cpu().numpy().copy(); s = res.boxes.conf.cpu().numpy(); c = res.boxes.cls.cpu().numpy().astype(int)
    if flip: b[:, [0, 2]] = 1 - b[:, [2, 0]]
    b[:, [0, 2]] = x0 + b[:, [0, 2]] * sx; b[:, [1, 3]] = y0 + b[:, [1, 3]] * sy
    return b, s, c

def raw_views(model, path, device, conf=0.03, iou=0.7):
    """한 이미지에 대한 여러 관점의 원시 예측 dict: full640/full960/full1280(+flip), tile2x2(960)."""
    im = Image.open(path).convert("RGB"); W, H = im.size; arr = np.asarray(im)[:, :, ::-1]  # BGR
    kw = dict(conf=conf, iou=iou, device=device, verbose=False, max_det=300); out = {}
    for sz in (640, 960, 1280):
        out[f"full{sz}"] = _collect(model.predict(arr, imgsz=sz, **kw)[0])
        out[f"flip{sz}"] = _collect(model.predict(arr[:, ::-1].copy(), imgsz=sz, **kw)[0], flip=True)
    # 2x2 타일(겹침 25%): 작은 객체용
    tw, th = int(W * 0.625), int(H * 0.625); parts = []
    for x in (0, W - tw):
        for y in (0, H - th):
            r = model.predict(arr[y:y + th, x:x + tw].copy(), imgsz=960, **kw)[0]
            parts.append(_collect(r, x / W, y / H, tw / W, th / H))
    out["tile960"] = tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
    return out

def _iou_inter(a, B):
    iw = np.maximum(0, np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0])); ih = np.maximum(0, np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]))
    inter = iw * ih; aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa + ab - inter + 1e-12), inter / (np.minimum(aa, ab) + 1e-12)

def merge(views, keys, group, nms_iou=0.5, contain=1.01, vote=1):
    """선택한 관점들의 박스를 합쳐 (묶음 안에서 클래스 구분 없이) 점수순 군집화.
    반환: 군집 대표 점수 배열. 점수 = 군집 안 관점별 최고 conf 의 합 / 관점 수. vote = 군집에 필요한 최소 관점 수.
    contain<=1 이면 작은 박스가 큰 박스에 그 비율 이상 포함될 때도 같은 군집으로 본다(중복·부분 박스 억제)."""
    B, S, V = [], [], []
    for vi, k in enumerate(keys):
        b, s, c = views[k]; m = np.isin(c, GROUPS[group]); B.append(b[m]); S.append(s[m]); V.append(np.full(int(m.sum()), vi))
    B = np.concatenate(B); S = np.concatenate(S); V = np.concatenate(V)
    order = np.argsort(-S); reps, members = [], []
    for i in order:
        placed = False
        if reps:
            io, ct = _iou_inter(B[i], B[reps])
            j = np.where((io >= nms_iou) | (ct >= contain))[0]
            if len(j): members[j[0]].append(i); placed = True
        if not placed: reps.append(i); members.append([i])
    scores = []
    for mem in members:
        best = {}
        for i in mem: best[V[i]] = max(best.get(V[i], 0), S[i])
        if len(best) >= vote: scores.append(sum(best.values()) / len(keys))
    return np.array(scores)

def count(scores, thr): return int((scores >= thr).sum())
