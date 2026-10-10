import os, glob, collections, json, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from PIL import Image
from ultralytics import YOLO
W = r"C:\Users\User\Desktop\root\docker11\notebooks\MAICON_hub\training_s\26_10_09\all_data\problem01\yolov8s.pt"
AO = "data/aerial_objects"; TL = "data/transfer_learning"; tf = sorted(f for f in os.listdir(AO) if f.endswith(".jpg"))
m = YOLO(W); names = m.names; dev = 0 if torch.cuda.is_available() else "cpu"
AIR, BIRD = 4, 14                                                   # COCO: airplane=4, bird=14
fn = "_p2_coco_pass.pkl"
if os.path.exists(fn): R = pickle.load(open(fn, "rb"))
else:
    R = {}
    for i in range(0, len(tf), 20):
        for f, r in zip(tf[i:i + 20], m.predict([os.path.join(AO, x) for x in tf[i:i + 20]], imgsz=640, conf=0.05, iou=0.7, device=dev, verbose=False)):
            R[f] = (r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())
    pickle.dump(R, open(fn, "wb"))
print("== 5. 사전학습 yolov8s(COCO)가 테스트 300장에서 보는 것 (imgsz 640, conf 0.05 이상) ==")
rows = []
for f in tf:
    c, s, b = R[f]; rows.append(dict(f=f, n_air=int(((c == AIR) & (s >= 0.25)).sum()), n_bird=int(((c == BIRD) & (s >= 0.25)).sum()), top_air=float(s[c == AIR].max()) if (c == AIR).any() else 0.0, top_bird=float(s[c == BIRD].max()) if (c == BIRD).any() else 0.0,
                     top_any=float(s.max()) if len(s) else 0.0, other=[names[k] for k in np.unique(c[(s >= 0.25) & (c != AIR) & (c != BIRD)])]))
D = pd.DataFrame(rows)
kind = np.where((D.n_air > 0) & (D.n_bird == 0), "airplane만", np.where((D.n_bird > 0) & (D.n_air == 0), "bird만", np.where((D.n_air > 0) & (D.n_bird > 0), "airplane+bird 둘 다", "airplane·bird 없음(conf 0.25)")))
print("이미지 분류(conf 0.25):", dict(pd.Series(kind).value_counts()))
print("airplane 수 분포:", D.n_air[D.n_air > 0].value_counts().sort_index().to_dict(), "| bird 수 분포:", D.n_bird[D.n_bird > 0].value_counts().sort_index().to_dict())
print("둘 다 검출된 이미지(지문: 이미지당 한 종류만 존재):", D[(D.n_air > 0) & (D.n_bird > 0)].f.tolist()[:10])
none = D[(D.n_air == 0) & (D.n_bird == 0)]; print("airplane·bird 없음 %d장 중 다른 COCO 클래스 최고 점수 분포: %s" % (len(none), np.percentile(none.top_any, [10, 50, 90]).round(2).tolist()))
print("  그 이미지들에서 많이 나온 클래스:", dict(collections.Counter(c for l in none.other for c in l).most_common(8)))
print("  airplane/bird 점수가 0.05~0.25 사이(경계)인 이미지 %d장" % ((((D.top_air > 0.05) & (D.top_air < 0.25)) | ((D.top_bird > 0.05) & (D.top_bird < 0.25))) & (D.n_air + D.n_bird == 0)).sum())
print("\n== 6. 테스트 이미지 크기 대 종류 (큰 이미지가 어느 쪽인지) ==")
meta = pd.read_csv("_p2_test_meta.csv"); D = D.merge(meta, on="f"); D["kind"] = kind
print(D.groupby("kind").agg(장수=("f", "size"), 해상도중앙=("mp", "median"), 종횡비중앙=("ar", "median")).round(2).to_string())
# 근접 이미지 묶음: 비슷한 쌍이 종류를 공유하는지
def ah(p):
    a = np.asarray(Image.open(p).convert("L").resize((8, 8), Image.LANCZOS), float); return (a > a.mean()).flatten()
H = {f: ah(os.path.join(AO, f)) for f in tf}; pairs = [(a, b) for i, a in enumerate(tf) for b in tf[i + 1:] if (H[a] != H[b]).sum() <= 2]
k = dict(zip(D.f, D.kind)); cnt = dict(zip(D.f, zip(D.n_air, D.n_bird)))
print("\n== 7. 비슷한 쌍 %d개: 종류가 같은 쌍 %d, 다른 쌍 %d | 수량까지 같은 쌍 %d ==" % (len(pairs), sum(k[a] == k[b] for a, b in pairs), sum(k[a] != k[b] for a, b in pairs), sum(cnt[a] == cnt[b] for a, b in pairs)))
print("  예:", [(a, b, k[a], k[b]) for a, b in pairs[:4]])
D.drop(columns=["other"]).to_csv("_p2_coco_pass.csv", index=False)
