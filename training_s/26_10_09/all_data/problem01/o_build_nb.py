import nbformat, os, shutil
from nbclient import NotebookClient

os.makedirs("versions", exist_ok=True)
if not os.path.exists("versions/notebook_before_confirmed.ipynb"): shutil.copy("work.ipynb", "versions/notebook_before_confirmed.ipynb")
nb = nbformat.read("versions/notebook_before_confirmed.ipynb", as_version=4)

md_head = nbformat.v4.new_markdown_cell("""## 풀이: YOLOv8n 전이학습 + 검출 개수 세기
- **방법**: 사전학습 `yolov8n.pt`를 제공 학습 이미지 640장으로 전이학습한 뒤, 테스트 이미지에서 검출된 박스를 클래스별로 세어 수량으로 제출합니다.
- **확정 설정**: 학습 imgsz 640, batch 16, 40에폭, seed 0. 추론 imgsz 640, NMS iou 0.6, `agnostic_nms=True`, conf 0.3.
- **conf 선택 근거**: 검증 160장에서 conf 0.15~0.6을 비교한 결과 0.3이 가장 높았습니다(아래 표). 검증 160장으로 conf를 고르고 점수도 같이 봤기 때문에 **낙관적인 추정**입니다.
- **정직한 기록**: 학습은 개발 PC GPU에서 별도 스크립트로 1회 했고, 그 가중치(`best.pt`)가 있으면 이 노트북은 재학습을 건너뛰고 그대로 불러옵니다(가중치가 없으면 같은 설정으로 학습). 서버(CPU 2코어)에서의 학습 시간은 측정하지 않았습니다.""")

c_setup = nbformat.v4.new_code_cell('''import sys, json, os, glob, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd

try:
    import ultralytics
except ImportError:
    # 응시 환경에 없으면 설치 (용량·시간은 14.6 예산에 포함). 설치 후 이 셀을 다시 실행
    !pip install -q ultralytics
    import ultralytics
import torch
from ultralytics import YOLO
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())

DEVICE = 0 if torch.cuda.is_available() else "cpu"        # GPU 가 없으면 CPU 로 같은 코드 실행
NAMES = ["military_tank", "military_truck", "military_vehicle", "civilian_vehicle"]
# 제공 data.yaml 의 경로는 대회 서버용(/workspace/...)이라, 현재 폴더 기준 절대경로로 새 yaml 을 만든다 (원본 data 는 수정하지 않음)
DATA_ABS = os.path.abspath("data")
with open("data_work.yaml", "w", encoding="utf-8") as f:
    f.write(f"path: {json.dumps(DATA_ABS)}\\ntrain: train/images\\nval: val/images\\nnames:\\n" + "".join(f"  {i}: {n}\\n" for i, n in enumerate(NAMES)))
print(open("data_work.yaml", encoding="utf-8").read())''')

c_train = nbformat.v4.new_code_cell('''# ---- 전이학습 (확정 설정). 확정 가중치가 이미 있으면 재학습을 건너뛴다 ----
CONFIRMED = ["runs/detect/training_runs/gpu640/weights/best.pt", "training_runs/gpu640/weights/best.pt"]
found = [p for p in CONFIRMED if os.path.exists(p)]
if found:
    WEIGHTS = found[0]
    print("확정 가중치 사용(재학습 건너뜀):", WEIGHTS)
else:
    t0 = time.time()
    kw = dict(time=0.4) if DEVICE == "cpu" else {}          # CPU 에서는 시간 상한(0.4시간)을 둔다 -> 확정 모델과 달라질 수 있음
    res = YOLO("yolov8n.pt").train(data="data_work.yaml", epochs=40, imgsz=640, batch=16, workers=0, device=DEVICE,
                                   project="training_runs", name="work_final", seed=0, deterministic=True, plots=False,
                                   verbose=False, patience=10, exist_ok=True, **kw)
    WEIGHTS = str(res.save_dir / "weights" / "best.pt")
    print(f"학습 완료 {time.time() - t0:.0f}s ->", WEIGHTS)''')

c_val = nbformat.v4.new_code_cell('''# ---- 검증 160장: 이미지별 4개 수량이 모두 맞아야 1점 (Exact Count Accuracy) ----
model = YOLO(WEIGHTS)
IMGSZ, IOU, AGNOSTIC = 640, 0.6, True

def gt_counts(img):
    p = Path(img); lab = p.parents[1] / "labels" / (p.stem + ".txt")
    c = [0] * 4
    for l in open(lab).read().splitlines():
        if l.strip(): c[int(l.split()[0])] += 1
    return tuple(c)

def predict_boxes(paths, bs=40):
    out = []
    for i in range(0, len(paths), bs):
        for r in model.predict(paths[i:i + bs], imgsz=IMGSZ, conf=0.05, iou=IOU, agnostic_nms=AGNOSTIC, device=DEVICE, verbose=False):
            out.append((r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()))
    return out

def counts(box, thr): return tuple(int(((box[0] == c) & (box[1] >= thr)).sum()) for c in range(4))

val_imgs = sorted(glob.glob("data/val/images/*")); G = [gt_counts(p) for p in val_imgs]; VB = predict_boxes(val_imgs)
print("검증 이미지:", len(val_imgs), "| 전부 0 으로 찍었을 때 정확도: %.4f" % np.mean([g == (0, 0, 0, 0) for g in G]))
tab = {t: np.mean([counts(b, t) == g for b, g in zip(VB, G)]) for t in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)}
print("conf 별 정확한 수량 일치율:", {k: round(v, 4) for k, v in tab.items()})
CONF = 0.3
print("선택 conf =", CONF, "-> 검증 일치율 %.4f (conf 를 같은 데이터로 골랐으므로 낙관적)" % tab[CONF])''')

c_sub = nbformat.v4.new_code_cell('''# ---- 테스트 이미지 수량 예측 -> submission.csv ----
sample = pd.read_csv("data/sample_submission.csv")
paths = {os.path.basename(p): p for p in glob.glob("data/test/images/*")}
assert set(sample.image_id) == set(paths), "test 이미지와 sample_submission ID 가 다름"
TB = predict_boxes([paths[i] for i in sample.image_id])
sub = pd.DataFrame([[i] + list(counts(b, CONF)) for i, b in zip(sample.image_id, TB)], columns=["image_id"] + NAMES)
assert sub.image_id.tolist() == sample.image_id.tolist() and list(sub.columns) == list(sample.columns)
assert (sub[NAMES] >= 0).all().all() and not sub.isna().any().any()
sub.to_csv("submission.csv", index=False)
print(sub.shape, "클래스별 합계:", sub[NAMES].sum().to_dict(), "| 모두 0인 이미지 비율 %.3f (검증 정답은 %.3f)" % ((sub[NAMES].sum(axis=1) == 0).mean(), np.mean([g == (0, 0, 0, 0) for g in G])))
sub.head()''')

nb.cells = nb.cells[:2] + [md_head, c_setup, c_train, c_val, c_sub] + nb.cells[3:]
for c in nb.cells:
    if c.cell_type == "code": c.outputs = []; c.execution_count = None
nb.cells[1].source = nb.cells[1].source          # 원래의 데이터 확인 셀은 그대로
NotebookClient(nb, timeout=3000, kernel_name="python3", resources={"metadata": {"path": os.getcwd()}}).execute()
nbformat.write(nb, "work.ipynb")
