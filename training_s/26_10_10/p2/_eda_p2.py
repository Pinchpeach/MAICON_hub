import os, glob, hashlib, collections, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image
D = "data"; TL = os.path.join(D, "transfer_learning"); AO = os.path.join(D, "aerial_objects")
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()

print("== 1. 전이학습 데이터 (이미지·라벨 짝, 클래스, 박스 크기) ==")
imgs = sorted(f for f in os.listdir(TL) if f.lower().endswith((".jpg", ".jpeg", ".png"))); txts = sorted(f for f in os.listdir(TL) if f.endswith(".txt"))
print("이미지 %d장, 라벨 %d개 | 이미지 이름 예: %s | 확장자: %s" % (len(imgs), len(txts), imgs[:3], dict(collections.Counter(os.path.splitext(f)[1] for f in imgs))))
st = {os.path.splitext(f)[0] for f in imgs}; lt = {os.path.splitext(f)[0] for f in txts}
print("라벨 없는 이미지:", sorted(st - lt)[:5], "| 이미지 없는 라벨:", sorted(lt - st)[:5])
rows = []; bad = []
for f in txts:
    L = [l.split() for l in open(os.path.join(TL, f), encoding="utf-8", errors="ignore").read().splitlines() if l.strip()]
    for r in L:
        if len(r) != 5: bad.append((f, r)); continue
        c, x, y, w, h = int(float(r[0])), *map(float, r[1:]); rows.append((f, c, x, y, w, h))
B = pd.DataFrame(rows, columns=["file", "cls", "x", "y", "w", "h"]); B["area"] = B.w * B.h
print("박스 %d개, 라벨 형식이 이상한 행 %d | 클래스 분포: %s" % (len(B), len(bad), B.cls.value_counts().to_dict()))
print("이미지당 박스 수:", B.groupby("file").size().value_counts().sort_index().to_dict(), "| 라벨 비어 있는 파일:", sum(os.path.getsize(os.path.join(TL, f)) == 0 for f in txts))
print("박스 면적 비율(이미지 대비): 최소 %.4f, 10%% %.4f, 중앙 %.3f, 90%% %.3f, 최대 %.3f" % (B.area.min(), B.area.quantile(.1), B.area.median(), B.area.quantile(.9), B.area.max()))
print("면적 5%% 미만(작은 드론) %d개, 50%% 이상 %d개 | 좌표 범위 벗어남 %d" % ((B.area < 0.05).sum(), (B.area >= 0.5).sum(), ((B.x - B.w / 2 < -0.01) | (B.x + B.w / 2 > 1.01) | (B.y - B.h / 2 < -0.01) | (B.y + B.h / 2 > 1.01)).sum()))
sz = [Image.open(os.path.join(TL, f)).size for f in imgs]; print("이미지 크기 상위:", dict(collections.Counter(sz).most_common(5)), "| 가로 범위 %d~%d, 세로 범위 %d~%d" % (min(s[0] for s in sz), max(s[0] for s in sz), min(s[1] for s in sz), max(s[1] for s in sz)))

print("\n== 2. 테스트 데이터 (aerial_objects) ==")
tf = sorted(f for f in os.listdir(AO) if f.lower().endswith((".jpg", ".jpeg", ".png"))); print("이미지 %d장 | 이름 예: %s ~ %s | 번호 연속: %s" % (len(tf), tf[0], tf[-1], [int(f[4:7]) for f in tf] == list(range(1, len(tf) + 1))))
tsz = [Image.open(os.path.join(AO, f)).size for f in tf]; print("이미지 크기 상위:", dict(collections.Counter(tsz).most_common(6)), "| 가로 %d~%d, 세로 %d~%d" % (min(s[0] for s in tsz), max(s[0] for s in tsz), min(s[1] for s in tsz), max(s[1] for s in tsz)))
bad_img = []
for f in tf:
    try: Image.open(os.path.join(AO, f)).verify()
    except Exception as e: bad_img.append(f)
print("열 수 없는 이미지:", bad_img)

print("\n== 3. 중복과 겹침 (내용 해시) ==")
th = {f: md5(os.path.join(AO, f)) for f in tf}; trh = {f: md5(os.path.join(TL, f)) for f in imgs}
g = collections.defaultdict(list)
for f, h in th.items(): g[h].append(f)
dup = [v for v in g.values() if len(v) > 1]; print("테스트 안의 바이트 동일 중복 묶음 %d개 (이미지 %d장이 중복에 속함, 고유 %d장)" % (len(dup), sum(len(v) for v in dup), len(g)))
print("  예:", dup[:4])
ov = [(f, [t for t, h2 in trh.items() if h2 == h]) for f, h in th.items() if h in set(trh.values())]; print("전이학습 이미지와 바이트 동일한 테스트 이미지 %d장" % len(ov), ov[:3])
# 시각적으로 비슷한 이미지(리사이즈·재압축) 탐지: 8x8 평균 해시
def ah(p):
    a = np.asarray(Image.open(p).convert("L").resize((8, 8), Image.LANCZOS), float); return (a > a.mean()).flatten()
A = {f: ah(os.path.join(AO, f)) for f in tf}; T = {f: ah(os.path.join(TL, f)) for f in imgs}
near_tt = [(a, b) for i, a in enumerate(tf) for b in tf[i + 1:] if (A[a] != A[b]).sum() <= 2 and th[a] != th[b]]
near_tl = [(a, b) for a in tf for b in imgs if (A[a] != T[b]).sum() <= 2 and th[a] != trh[b]]
print("테스트끼리 비슷하지만 바이트는 다른 쌍 %d개 | 전이학습과 비슷한 테스트 %d쌍" % (len(near_tt), len(near_tl)), near_tl[:3])

print("\n== 4. 테스트 이미지 밝기·종횡비 (분포 확인용) ==")
meta = pd.DataFrame({"f": tf, "w": [s[0] for s in tsz], "h": [s[1] for s in tsz]}); meta["ar"] = meta.w / meta.h; meta["mp"] = meta.w * meta.h / 1e6
print(meta[["w", "h", "ar", "mp"]].describe().loc[["min", "25%", "50%", "75%", "max"]].round(2).to_string())
tm = pd.DataFrame({"w": [s[0] for s in sz], "h": [s[1] for s in sz]}); tm["mp"] = tm.w * tm.h / 1e6; print("전이학습 이미지 해상도(메가픽셀): 중앙 %.2f, 최대 %.2f | 테스트 중앙 %.2f, 최대 %.2f" % (tm.mp.median(), tm.mp.max(), meta.mp.median(), meta.mp.max()))
B.to_csv("_p2_boxes.csv", index=False); meta.to_csv("_p2_test_meta.csv", index=False)
