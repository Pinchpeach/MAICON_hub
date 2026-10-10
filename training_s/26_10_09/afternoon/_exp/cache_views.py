import sys, pickle, time, pandas as pd
sys.path.insert(0, "_exp")
from lib import raw_views
from ultralytics import YOLO
p = sys.argv[1]; m = YOLO(f"{p}/yolov8n.pt"); out = {}
for split, csv in (("train_images", "train_labels.csv"), ("test_images", "sample_submission.csv")):
    t = time.time()
    for f in pd.read_csv(f"{p}/data/{csv}").filename: out[(split, f)] = raw_views(m, f"{p}/data/{split}/{f}", 0)
    print(p, split, "%.1fs" % (time.time() - t))
pickle.dump(out, open(f"_exp/{p}_views.pkl", "wb"))
