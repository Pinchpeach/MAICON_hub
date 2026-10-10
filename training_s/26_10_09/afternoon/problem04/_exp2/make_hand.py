"""이미지를 직접 보고 고친 박스 데이터셋. 모델 후보 중에서 고르고(keep/add/drop), 모델이 못 잡은 객체는 좌표를 직접 정해 추가(hand).
conf: 'sure' = 눈으로 분명히 확인, 'guess' = 정답 수량에 맞추려고 고른 추정(틀릴 수 있음)."""
import sys, os, json, shutil; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from harness import labels, DATA, ROOT
import make_dummy as M

OUT = os.path.join(ROOT, "_exp2", "dummy_boxes_hand")
for d in ("annotated", "images", "labels"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
C = json.load(open(os.path.join(ROOT, "_exp2", "review", "candidates.json")))
# 이미지별 수정. drop/add 는 후보 번호, hand 는 (그룹, [x0,y0,x1,y1] 전체 기준 0~1, 확신, 설명)
FIX = {
 "S12_2_0001.jpg": dict(hand=[("vehicle", [0.0, 0.165, 0.055, 0.27], "guess", "왼쪽 창 너머 길가의 차"), ("vehicle", [0.21, 0.19, 0.30, 0.235], "guess", "창에 비친 흰 차량")]),
 "S12_2_0004.jpg": dict(drop=["P1"], hand=[("person", [0.765, 0.03, 0.86, 0.455], "guess", "배경에 선 사람(왼쪽)"), ("person", [0.85, 0.03, 0.94, 0.455], "guess", "배경에 선 사람(오른쪽)")], note="모델은 흐린 두 사람을 한 박스로 봄"),
 "S12_2_0005.jpg": dict(drop=[("V4", "guess", "먼 배경의 버스로 보이나 정답 3대에 맞추려 제외")]),
 "S12_2_0006.jpg": dict(hand=[("vehicle", [0.19, 0.215, 0.755, 0.635], "sure", "비 맞은 유리 너머의 흰 승용차")]),
 "S12_2_0007.jpg": dict(add=[("P7", "guess", "오른쪽 위 가장자리의 머리"), ("P8", "guess", "은색 차 안의 동승자"), ("V8", "sure", "왼쪽 위 캠핑카")], drop=[("V5", "guess", "사람들 뒤 영역, 차량 불분명")]),
 "S12_2_0009.jpg": dict(hand=[("person", [0.62, 0.83, 1.0, 1.0], "guess", "아래쪽에서 염소를 만지는 손과 초록 소매")]),
 "S12_2_0010.jpg": dict(drop=[("P2", "guess", "트럭 유리 너머로 보이는 형체. 정답 1명에 맞추려 제외")]),
 "S12_2_0012.jpg": dict(hand=[("vehicle", [0.0, 0.0, 1.0, 1.0], "sure", "사진 전체가 차량(창 너머로 본 실내)")]),
 "S12_2_0013.jpg": dict(drop=[("P4", "sure", "포수 미트·무릎 부분, 사람 아님")], hand=[("person", [0.02, 0.17, 0.21, 0.62], "sure", "울타리 뒤 왼쪽의 빨간 셔츠 남자"), ("vehicle", [0.20, 0.03, 0.53, 0.19], "guess", "울타리 뒤 흐린 차량")]),
 "S12_2_0014.jpg": dict(note="차량 정답 2대 중 1대(오른쪽 아래 밴)만 확인. 나머지 1대는 찾지 못함"),
 "S12_2_0015.jpg": dict(add=[("V7", "guess", "오른쪽 차들 사이의 후미등")]),
 "S12_2_0016.jpg": dict(add=[("P3", "guess", "오른쪽 가장자리에 걸친 사람")], drop=[("V3", "sure", "파란 트럭의 뒷부분(같은 트럭을 둘로 셈)"), ("V4", "sure", "파란 트럭의 운전석")],
                        hand=[("vehicle", [0.61, 0.42, 0.90, 0.555], "sure", "파란 트럭 전체"), ("vehicle", [0.60, 0.495, 0.78, 0.565], "sure", "사고 차량 뒤의 흰 승용차")]),
 "S12_2_0017.jpg": dict(drop=[("V3", "guess", "왼쪽 가장자리의 좁은 조각")], add=[("P3", "guess", "은색 차의 운전자")]),
 "S12_2_0019.jpg": dict(hand=[("vehicle", [0.165, 0.655, 0.195, 0.69], "guess", "멀리 전조등 한 쌍")]),
 "S12_2_0020.jpg": dict(add=[("V2", "sure", "울타리 뒤의 어두운 승용차")]),
 "S12_2_0021.jpg": dict(drop=[("P4", "sure", "말 등 위의 작은 흰 물체, 사람 아님")]),
 "S12_2_0022.jpg": dict(hand=[("vehicle", [0.0, 0.73, 0.10, 0.96], "guess", "왼쪽 가장자리에 걸친 차")]),
 "S12_2_0028.jpg": dict(drop=[("P1", "guess", "아래 가장자리에 머리만 보이는 실제 사람이지만 정답은 0명")], hand=[("vehicle", [0.86, 0.885, 1.0, 0.975], "sure", "오른쪽 아래 표지판 뒤의 회색 밴")]),
 "S12_2_0030.jpg": dict(drop=[("P8", "guess", "오른쪽 아래 인형 머리")]),
}
norm = lambda items: [(x, "sure", "") if isinstance(x, str) else x for x in items]
rows, log = [], []
for n, f in enumerate(labels.filename):
    fx = FIX.get(f, {}); cand = {c["id"]: c for c in C.get(f, [])}
    if f not in C:                                                            # 수량이 맞았던 11장: 모델 박스 그대로
        cand = {}
        for g, pre in (("person", "P"), ("vehicle", "V")):
            for k, (b, s, kk, nv) in enumerate(sorted([c for c in M.clusters(f, g) if c[1] >= 0.03], key=lambda c: -c[1]), 1):
                cand[f"{pre}{k}"] = dict(id=f"{pre}{k}", group=g, box=[float(v) for v in b], score=float(s), kept=bool(s >= M.CFG[g]["thr"]))
    drops = {d[0]: d for d in norm(fx.get("drop", []))}; adds = {a[0]: a for a in norm(fx.get("add", []))}; boxes = []
    for cid, c in cand.items():
        if c["kept"] and cid not in drops: boxes.append((c["group"], c["box"], "model", "sure", cid))
        elif cid in adds: boxes.append((c["group"], c["box"], "picked", adds[cid][1], cid)); log.append((f, "후보 채택", cid, adds[cid][1], adds[cid][2]))
    for cid, d in drops.items(): log.append((f, "모델 박스 제외", cid, d[1], d[2]))
    for g, b, cf, why in fx.get("hand", []): boxes.append((g, b, "hand", cf, "H")); log.append((f, "직접 그림", g, cf, why))
    img = Image.open(os.path.join(DATA, "train_images", f)).convert("RGB"); W, H = img.size; sc = max(1.0, 900 / max(W, H)); can = img.resize((int(W * sc), int(H * sc)), Image.LANCZOS); cw, ch = can.size; dr = ImageDraw.Draw(can); fs = max(13, int(cw / 55)); ft = M.font(fs); lines = []
    for g, b, src, cf, cid in boxes:
        x0, y0, x1, y1 = b[0] * cw, b[1] * ch, b[2] * cw, b[3] * ch; col = M.COL[g]
        dr.rectangle([x0, y0, x1, y1], outline=col, width=3)
        if src != "model" and x1 - x0 > 12 and y1 - y0 > 12: dr.rectangle([x0 + 4, y0 + 4, x1 - 4, y1 - 4], outline=(255, 255, 255), width=1)       # 손으로 고른/그린 박스는 안쪽에 흰 선을 한 번 더
        tag = ("사람" if g == "person" else "차량") + {"model": "", "picked": " +후보", "hand": " +직접"}[src] + ("?" if cf == "guess" else ""); tb = dr.textbbox((0, 0), tag, font=ft); tw, th = tb[2] + 6, tb[3] + 6; ty = y0 - th if y0 - th >= 0 else y0
        dr.rectangle([x0, ty, x0 + tw, ty + th], fill=col); dr.text((x0 + 3, ty + 1), tag, fill=(255, 255, 255), font=ft)
        lines.append(f"{0 if g == 'person' else 1} {(b[0] + b[2]) / 2:.6f} {(b[1] + b[3]) / 2:.6f} {b[2] - b[0]:.6f} {b[3] - b[1]:.6f}")
    yp, yv = int(labels.person_count[n]), int(labels.vehicle_count[n]); pp = sum(b[0] == "person" for b in boxes); pv = sum(b[0] == "vehicle" for b in boxes); ok = pp == yp and pv == yv
    ng = sum(b[3] == "guess" for b in boxes) + sum(d[1] == "guess" for d in drops.values())
    head = Image.new("RGB", (cw, fs * 2 + 14), (252, 252, 251) if ok else (255, 236, 228)); ImageDraw.Draw(head).text((8, 5), f"{f}  정답: 사람 {yp} · 차량 {yv} | 박스: 사람 {pp} · 차량 {pv}  {'일치' if ok else '불일치'} | 흰 속선 = 직접 고르거나 그린 박스, ? = 추정", fill=(11, 11, 11), font=M.font(max(12, fs - 1)))
    full = Image.new("RGB", (cw, ch + head.size[1]), (252, 252, 251)); full.paste(head, (0, 0)); full.paste(can, (0, head.size[1])); full.save(os.path.join(OUT, "annotated", f), quality=92)
    shutil.copy(os.path.join(DATA, "train_images", f), os.path.join(OUT, "images", f)); open(os.path.join(OUT, "labels", f[:-4] + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
    rows.append(dict(filename=f, reviewed=f in FIX, person_true=yp, person_boxes=pp, vehicle_true=yv, vehicle_boxes=pv, both_match=ok, n_model=sum(b[2] == "model" for b in boxes), n_picked=sum(b[2] == "picked" for b in boxes), n_hand=sum(b[2] == "hand" for b in boxes), n_dropped=len(drops), n_guess=ng, note=fx.get("note", "")))
S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "summary.csv"), index=False); Lg = pd.DataFrame(log, columns=["filename", "action", "target", "confidence", "reason"]); Lg.to_csv(os.path.join(OUT, "changes.csv"), index=False, encoding="utf-8-sig")
open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8").write("# 사람이 직접 고친 박스(학습 30장). 'guess' 표시 박스는 추정임\npath: .\ntrain: images\nval: images\nnames:\n  0: person\n  1: vehicle\n")
th = [Image.open(os.path.join(OUT, "annotated", f)) for f in labels.filename]; cw_, ch_ = 520, 420; sheet = Image.new("RGB", (cw_ * 6, ch_ * 5), (252, 252, 251))
for i, im in enumerate(th):
    r = min(cw_ / im.size[0], ch_ / im.size[1]); t = im.resize((int(im.size[0] * r), int(im.size[1] * r)), Image.LANCZOS); sheet.paste(t, ((i % 6) * cw_ + (cw_ - t.size[0]) // 2, (i // 6) * ch_ + (ch_ - t.size[1]) // 2))
sheet.save(os.path.join(OUT, "contact_sheet.jpg"), quality=88)
print("\n=== 수정본 요약 ===")
print("수량 일치 %d/30 (수정 전 11/30) | 사람 박스 %d (정답 %d) | 차량 박스 %d (정답 %d)" % (S.both_match.sum(), S.person_boxes.sum(), S.person_true.sum(), S.vehicle_boxes.sum(), S.vehicle_true.sum()))
print("박스 출처: 모델 그대로 %d, 후보에서 채택 %d, 직접 그림 %d | 제외한 모델 박스 %d" % (S.n_model.sum(), S.n_picked.sum(), S.n_hand.sum(), S.n_dropped.sum()))
print("변경 %d건 중 확실 %d, 추정 %d" % (len(Lg), (Lg.confidence == "sure").sum(), (Lg.confidence == "guess").sum()))
print("아직 불일치:", S[~S.both_match][["filename", "person_true", "person_boxes", "vehicle_true", "vehicle_boxes", "note"]].to_string(index=False))
