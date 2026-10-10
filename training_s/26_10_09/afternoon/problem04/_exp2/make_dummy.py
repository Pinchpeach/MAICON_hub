"""학습 30장에 메인 설정의 탐지 박스를 그려 넣은 더미 데이터셋 생성. 박스는 사람이 그린 정답이 아니라 yolov8n 예측(의사 박스)이다."""
import sys, os, shutil; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch, torchvision
from PIL import Image, ImageDraw, ImageFont
from harness import labels, DATA, ROOT, cache, _iou_inter

OUT = os.path.join(ROOT, "_exp2", "dummy_boxes")
for d in ("annotated", "images", "labels"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
R = cache("train"); files = list(labels.filename)
CFG = {"person": dict(views=[(s, f) for s in (640, 960, 1280) for f in (False, True)], nms_iou=0.5, contain=0.8, vote=1, min_ratio=0.0, thr=0.15),
       "vehicle": dict(views=[(s, False) for s in (640, 960, 1280)], nms_iou=0.5, contain=0.7, vote=2, min_ratio=0.1, thr=0.175)}
VNAME = ["car", "bus", "truck"]; COL = {"person": (42, 120, 214), "vehicle": (235, 104, 52)}; GRAY = (150, 150, 146); LOWCONF = 0.05


def view_dets(view, group):
    b, c = view
    if group == "person": s = c[:, 0]; k = np.zeros(len(s), int)
    else: s = c[:, 1:4].max(1); k = c[:, 1:4].argmax(1)
    m = s >= 0.03; b, s, k = b[m], s[m], k[m]
    if not len(s): return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    keep = torchvision.ops.nms(torch.from_numpy(b).float(), torch.from_numpy(s).float(), 0.7).numpy()[:300]; return b[keep], s[keep], k[keep]


def clusters(f, group):
    """메인과 같은 군집 규칙. 반환: [(대표 박스 xyxy 정규화, 군집 점수, 세부 클래스 번호, 본 관점 수)]"""
    c = CFG[group]; D = [view_dets(R[f][v], group) for v in c["views"]]
    B = np.concatenate([d[0] for d in D]); S = np.concatenate([d[1] for d in D]); K = np.concatenate([d[2] for d in D]); V = np.concatenate([np.full(len(d[1]), i) for i, d in enumerate(D)])
    if not len(S): return []
    A = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]); reps, members = [], []
    for i in np.argsort(-S):
        if reps:
            io, ct = _iou_inter(B[i], B[reps]); ratio = np.minimum(A[i], A[reps]) / (np.maximum(A[i], A[reps]) + 1e-12)
            j = np.where((io >= c["nms_iou"]) | ((ct >= c["contain"]) & (ratio >= c["min_ratio"])))[0]
            if len(j): members[j[0]].append(i); continue
        reps.append(i); members.append([i])
    out = []
    for r, mem in zip(reps, members):
        best = {}
        for i in mem: best[V[i]] = max(best.get(V[i], 0), S[i])
        if len(best) >= c["vote"]: out.append((np.clip(B[r], 0, 1), sum(best.values()) / len(c["views"]), int(K[r]), len(best)))
    return out


def font(size):
    for p in ("C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


rows = []
for n, f in enumerate(files):
    img = Image.open(os.path.join(DATA, "train_images", f)).convert("RGB"); W, H = img.size; sc = max(1.0, 900 / max(W, H))      # 작은 이미지는 키워서 글자가 보이게
    can = img.resize((int(W * sc), int(H * sc)), Image.LANCZOS); cw, ch = can.size; dr = ImageDraw.Draw(can); fs = max(13, int(cw / 55)); ft = font(fs); lines = []; cnt = {}
    for gi, g in enumerate(("person", "vehicle")):
        cl = clusters(f, g); kept = [c for c in cl if c[1] >= CFG[g]["thr"]]; low = [c for c in cl if LOWCONF <= c[1] < CFG[g]["thr"]]; cnt[g] = (len(kept), len(low))
        for b, s, k, nv in low:                                                         # 임계값 아래 후보: 회색 가는 선 (세지 않음)
            x0, y0, x1, y1 = b[0] * cw, b[1] * ch, b[2] * cw, b[3] * ch; dr.rectangle([x0, y0, x1, y1], outline=GRAY, width=1)
            dr.text((x0 + 2, y1 - fs - 2), f"{'p' if g == 'person' else 'v'}? {s:.2f}", fill=GRAY, font=font(max(10, fs - 3)))
        for b, s, k, nv in kept:                                                        # 센 박스
            x0, y0, x1, y1 = b[0] * cw, b[1] * ch, b[2] * cw, b[3] * ch; dr.rectangle([x0, y0, x1, y1], outline=COL[g], width=3)
            tag = f"{'person' if g == 'person' else VNAME[k]} {s:.2f}"; tb = dr.textbbox((0, 0), tag, font=ft); tw, th = tb[2] - tb[0] + 6, tb[3] - tb[1] + 6; ty = y0 - th if y0 - th >= 0 else y0
            dr.rectangle([x0, ty, x0 + tw, ty + th], fill=COL[g]); dr.text((x0 + 3, ty + 1), tag, fill=(255, 255, 255), font=ft)
            xc, yc, bw, bh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]; lines.append(f"{gi} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    yp, yv = int(labels.person_count[n]), int(labels.vehicle_count[n]); pp, pv = cnt["person"][0], cnt["vehicle"][0]; ok = (pp == yp) and (pv == yv)
    head = Image.new("RGB", (cw, fs * 2 + 14), (252, 252, 251) if ok else (255, 236, 228)); hd = ImageDraw.Draw(head)
    hd.text((8, 5), f"{f}   정답: 사람 {yp} · 차량 {yv}   |   박스: 사람 {pp} · 차량 {pv}   {'일치' if ok else '불일치'}   (회색 = 임계값 아래 후보, 세지 않음)", fill=(11, 11, 11), font=font(max(12, fs - 1)))
    full = Image.new("RGB", (cw, ch + head.size[1]), (252, 252, 251)); full.paste(head, (0, 0)); full.paste(can, (0, head.size[1])); full.save(os.path.join(OUT, "annotated", f), quality=92)
    shutil.copy(os.path.join(DATA, "train_images", f), os.path.join(OUT, "images", f)); open(os.path.join(OUT, "labels", f.rsplit(".", 1)[0] + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
    rows.append(dict(filename=f, width=W, height=H, person_true=yp, person_boxes=pp, person_diff=pp - yp, person_low_candidates=cnt["person"][1], vehicle_true=yv, vehicle_boxes=pv, vehicle_diff=pv - yv, vehicle_low_candidates=cnt["vehicle"][1], both_match=ok))
S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8").write("# 의사 박스(yolov8n 예측)로 만든 더미 데이터셋. 사람이 그린 정답이 아님\npath: .\ntrain: images\nval: images\nnames:\n  0: person\n  1: vehicle\n")
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(f"""# 문제 4 더미 박스 데이터셋 (학습 30장)

- **박스는 정답이 아닙니다.** COCO 사전학습 yolov8n 의 예측을 메인 설정(사람: 640·960·1280 + 좌우 반전, 점수 0.15 이상 / 차량: 640·960·1280, 2관점 이상, 점수 0.175 이상)으로 합친 의사 박스입니다.
- `annotated/` 박스를 그려 넣은 이미지(파랑 = 사람, 주황 = 차량, 회색 가는 선 = 점수 0.05 이상이지만 임계값 아래라 세지 않은 후보). 위쪽 띠에 정답 수량과 박스 수를 적었고, 불일치하면 띠가 붉은색입니다.
- `images/` 원본 복사본, `labels/` YOLO 형식 라벨(`class x_center y_center width height`, 0 = person, 1 = vehicle), `data.yaml`.
- `summary.csv` 이미지별 정답 수량, 박스 수, 차이, 임계값 아래 후보 수.
- 박스 수가 정답과 모두 일치하는 이미지: {int(S.both_match.sum())}/30. 사람 박스 합 {int(S.person_boxes.sum())} (정답 {int(S.person_true.sum())}), 차량 박스 합 {int(S.vehicle_boxes.sum())} (정답 {int(S.vehicle_true.sum())}).
- 생성 스크립트: `_exp2/make_dummy.py`
""")
# 한눈에 보는 모음 그림 (6 x 5)
th = [Image.open(os.path.join(OUT, "annotated", f)) for f in files]; cw_, ch_ = 520, 420; sheet = Image.new("RGB", (cw_ * 6, ch_ * 5), (252, 252, 251))
for i, im in enumerate(th):
    r = min(cw_ / im.size[0], ch_ / im.size[1]); t = im.resize((int(im.size[0] * r), int(im.size[1] * r)), Image.LANCZOS); sheet.paste(t, ((i % 6) * cw_ + (cw_ - t.size[0]) // 2, (i // 6) * ch_ + (ch_ - t.size[1]) // 2))
sheet.save(os.path.join(OUT, "contact_sheet.jpg"), quality=88)
print(S[["filename", "person_true", "person_boxes", "vehicle_true", "vehicle_boxes", "both_match"]].to_string(index=False))
print("일치 %d/30 | 사람 박스 합 %d (정답 %d) | 차량 박스 합 %d (정답 %d) | 라벨 행 수 %d" % (S.both_match.sum(), S.person_boxes.sum(), S.person_true.sum(), S.vehicle_boxes.sum(), S.vehicle_true.sum(), S.person_boxes.sum() + S.vehicle_boxes.sum()))
