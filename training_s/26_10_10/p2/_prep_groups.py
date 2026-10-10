"""비슷한 이미지를 한 묶음으로 보고 5겹 그룹 교차검증 분할을 만든다."""
import os, json, numpy as np
from PIL import Image
SRC = "data/transfer_learning"; imgs = sorted(f for f in os.listdir(SRC) if f.lower().endswith((".jpg", ".jpeg", ".png")))
def ah(p, n=8):
    a = np.asarray(Image.open(p).convert("L").resize((n, n), Image.LANCZOS), float); return (a > a.mean()).flatten()
H = {f: ah(os.path.join(SRC, f)) for f in imgs}; par = {f: f for f in imgs}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
TH = 6; links = []
for i, a in enumerate(imgs):
    for b in imgs[i + 1:]:
        d = int((H[a] != H[b]).sum())
        if d <= TH: links.append((a, b, d)); par[find(a)] = find(b)
G = {}
for f in imgs: G.setdefault(find(f), []).append(f)
groups = sorted(G.values(), key=lambda g: (-len(g), g[0])); print("해시 거리 <= %d 로 묶은 결과: 이미지 %d장 -> 묶음 %d개 (크기 2 이상 %d개)" % (TH, len(imgs), len(groups), sum(len(g) > 1 for g in groups)))
for g in groups:
    if len(g) > 1: print("  묶음:", g)
folds = [[] for _ in range(5)]
for g in groups:                                   # 큰 묶음부터 가장 작은 겹에 배정(균형)
    min(folds, key=len).extend(g)
print("겹 크기:", [len(f) for f in folds])
json.dump({"folds": [sorted(f) for f in folds], "links": links}, open("_folds.json", "w"), ensure_ascii=False, indent=0)
