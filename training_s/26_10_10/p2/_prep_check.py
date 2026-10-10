import os, collections, warnings; warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
TL = "data/transfer_learning"; AO = "data/aerial_objects"
tr = set(os.listdir("_tl_data/images/train")); va = set(os.listdir("_tl_data/images/val"))
print("== 1. 이미지 모드 (PNG 투명도·팔레트 여부) ==")
for nm, d, fs in (("전이학습", TL, sorted(f for f in os.listdir(TL) if f.lower().endswith((".jpg", ".jpeg", ".png")))), ("테스트", AO, sorted(os.listdir(AO)))):
    c = collections.Counter(); alpha = []
    for f in fs:
        im = Image.open(os.path.join(d, f)); c[(os.path.splitext(f)[1].lower(), im.mode)] += 1
        if im.mode in ("RGBA", "LA", "P") and ("transparency" in im.info or im.mode in ("RGBA", "LA")): alpha.append(f)
    print("  %s: %s | 투명도가 있는 이미지 %d장 %s" % (nm, dict(c), len(alpha), alpha[:6]))
print("\n== 2. 전이학습 58장 안의 비슷한 이미지 (8x8 평균 해시 거리 <= 4) ==")
def ah(p):
    a = np.asarray(Image.open(p).convert("L").resize((8, 8), Image.LANCZOS), float); return (a > a.mean()).flatten()
fs = sorted(os.listdir("_tl_data/images/all")); H = {f: ah(os.path.join("_tl_data/images/all", f)) for f in fs}
pairs = [(a, b, int((H[a] != H[b]).sum())) for i, a in enumerate(fs) for b in fs[i + 1:] if (H[a] != H[b]).sum() <= 4]
cross = [(a, b, d) for a, b, d in pairs if (a in va) != (b in va)]
print("  비슷한 쌍 %d개 | 그중 학습과 검증에 걸친 쌍 %d개 (검증 점수가 부풀 수 있음)" % (len(pairs), len(cross)))
for a, b, d in cross[:10]: print("     %s(%s) - %s(%s) 거리 %d" % (a, "검증" if a in va else "학습", b, "검증" if b in va else "학습", d))
vcross = sorted({x for a, b, d in cross for x in (a, b) if x in va}); print("  영향받는 검증 이미지 %d / 11장: %s" % (len(vcross), vcross))
print("\n== 3. 테스트와 비슷한 전이학습 이미지 (학습에 쓰면 테스트 일부는 이미 본 이미지) ==")
tf = sorted(os.listdir(AO)); T = {f: ah(os.path.join(AO, f)) for f in tf}
near = [(t, f, int((T[t] != H[f]).sum())) for t in tf for f in fs if (T[t] != H[f]).sum() <= 4]; print("  비슷한 (테스트, 전이학습) 쌍 %d개:" % len(near), near[:8])
print("\n== 4. 이미지 크기와 YOLO 입력 ==")
sz = [Image.open(os.path.join(TL, f)).size for f in fs]; mx = [max(s) for s in sz]
print("  전이학습 긴 변: 중앙 %d, 최대 %d | 640 초과 %d장 → imgsz 640 에서 축소됨(작은 드론은 더 작아짐)" % (np.median(mx), max(mx), sum(m > 640 for m in mx)))
print("  박스를 640 으로 줄였을 때 가로·세로 최소 변(픽셀) 하위:", sorted(round(min(w, h) * 640 / max(w, h) * 1.0, 0) for w, h in sz)[:3], "(이미지 전체 기준. 박스는 이보다 작음)")
