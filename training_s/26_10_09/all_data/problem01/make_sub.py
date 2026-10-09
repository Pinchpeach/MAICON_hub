import glob, os, sys, pandas as pd
from ultralytics import YOLO
w, imgsz, iou, conf, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
sample = pd.read_csv("data/sample_submission.csv"); names = ["military_tank", "military_truck", "military_vehicle", "civilian_vehicle"]
paths = {os.path.basename(p): p for p in glob.glob("data/test/images/*")}
assert set(sample.image_id) == set(paths), "test 이미지와 sample ID 불일치"
model = YOLO(w); rows = []
for i in range(0, len(sample), 50):
    ids = list(sample.image_id[i:i + 50])
    for iid, r in zip(ids, model.predict([paths[x] for x in ids], imgsz=imgsz, conf=conf, iou=iou, device=0, verbose=False)):
        k = r.boxes.cls.cpu().numpy().astype(int); rows.append([iid] + [int((k == c).sum()) for c in range(4)])
sub = pd.DataFrame(rows, columns=["image_id"] + names)
assert sub.image_id.tolist() == sample.image_id.tolist() and list(sub.columns) == list(sample.columns)
assert (sub[names] >= 0).all().all() and not sub.isna().any().any()
sub.to_csv(out, index=False)
print(sub.shape, "합계 수량:", sub[names].sum().to_dict(), "| 모두 0인 이미지 비율 %.3f" % (sub[names].sum(1) == 0).mean())
