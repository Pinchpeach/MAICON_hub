CELL_NEG = '''# ---- 1) 기본 yolov8s 로 테스트 300장을 먼저 추론하고, 확신하는 새·비행기 이미지를 '드론 없음' 부정 예시로 고른다 ----
# 사용 범위: 테스트 이미지의 입력만 사용하고 정답은 쓰지 않는다. 가짜 라벨은 기본 모델의 확신 예측이다.
AIR, BIRD = 4, 14
test_files = sorted(f for f in os.listdir(AO) if f.lower().endswith(".jpg")); assert len(test_files) == 300
def predict_all(weights, conf=0.05, iou=0.6, bs=10):
    m = YOLO(weights); out = {}
    for i in range(0, len(test_files), bs):
        for f, r in zip(test_files[i:i + bs], m.predict([os.path.join(AO, x) for x in test_files[i:i + bs]], imgsz=640, conf=conf, iou=iou, device=DEVICE, verbose=False)):
            out[f] = (r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())
    return out
t0 = time.time(); P_base = predict_all("yolov8s.pt"); t_base = time.time() - t0
def base_scores(f):
    c, s, b = P_base[f]; W, H = Image.open(os.path.join(AO, f)).size
    sa = s[c == AIR].max() if (c == AIR).any() else 0.0; sb = s[c == BIRD].max() if (c == BIRD).any() else 0.0; ar = np.nan
    if sa > 0:
        i = np.where(c == AIR)[0][np.argmax(s[c == AIR])]; ar = (b[i][2] - b[i][0]) * W / max((b[i][3] - b[i][1]) * H, 1e-6)
    return sa, sb, ar
SC = {f: base_scores(f) for f in test_files}
neg_bird = [f for f in test_files if SC[f][1] >= 0.7 and SC[f][0] < 0.3]                                  # 새로 확신(점수 0.7 이상), 비행기 아님
neg_air = [f for f in test_files if SC[f][0] >= 0.8 and SC[f][2] >= 3.0 and SC[f][1] < 0.3]               # 옆으로 긴 비행기로 확신(점수 0.8 이상, 가로/세로 3.0 이상)
NEG = neg_bird + neg_air
print(f"기본 모델 추론 {t_base:.0f}s | 부정 예시: 새 {len(neg_bird)}장, 비행기 {len(neg_air)}장, 합 {len(NEG)}장 (300장 중)")
'''

CELL_C2 = '''# ---- 2) 드론 전이학습: 드론 58장(회전·반전·합성 포함) + 부정 예시, yolov8s, 학습률 0.0002, 20에폭 ----
yml, n_pos = build_train_set()
nd = f"{WORK}/dataset/train"
for f in NEG:
    im = Image.open(os.path.join(AO, f)).convert("RGB"); w, h = im.size; s = MAXS / max(w, h)
    if s < 1: im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    stem = "neg_" + os.path.splitext(f)[0]; im.save(f"{nd}/images/{stem}.jpg", quality=92); open(f"{nd}/labels/{stem}.txt", "w").write("")     # 빈 라벨 = 드론 없음
n_train = len(glob.glob(f"{nd}/images/*"))
kw = dict(epochs=20, imgsz=640, batch=8, optimizer="AdamW", lr0=0.0002, patience=0, seed=0, deterministic=True, workers=0, plots=False, verbose=False, exist_ok=True, cache="ram",
          project=os.path.abspath(f"{WORK}/runs"), name="drone_s_neg")
if DEVICE == "cpu": kw["time"] = 0.4                     # CPU 에서는 시간 상한(0.4시간)을 둔다
t0 = time.time(); drone_model = YOLO("yolov8s.pt"); res = drone_model.train(data=yml, device=DEVICE, **kw); t_train = time.time() - t0
DRONE_W = os.path.join(str(drone_model.trainer.save_dir), "weights", "last.pt")
print(f"학습 이미지 {n_train}장(드론 {n_pos}장 + 부정 {len(NEG)}장), 학습 {t_train:.0f}s ({t_train / kw['epochs']:.1f}s/에폭, {DEVICE}) | 가중치: {DRONE_W}")
'''

CELL_D2 = '''# ---- 3) 드론 모델로 테스트 300장 추론 ----
t0 = time.time(); P_drone = predict_all(DRONE_W); t_pred = time.time() - t0
print(f"드론 모델 추론 {t_pred:.0f}s ({DEVICE})")
'''

CELL_E2 = '''# ---- 4) 종류 판정과 수량, submission.csv ----
import collections
TAU = 0.4      # 드론 판정 임계값: 보류한 드론 12장 재현율 1.00, 보류한 부정 27장 오탐 0 을 함께 만족하는 가장 높은 값(겹 0 학습 기준)
TD = 0.35      # 드론 개수를 셀 때 박스 임계값(그룹 교차검증에서 0.3~0.4 가 비슷)
TP, TP_LO = 0.5, 0.1

def iou_contain(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-12), inter / (min(aa, ab) + 1e-12)

def merge_boxes(boxes, scores, thr, iou_t=0.5, contain_t=0.8):
    """점수 thr 이상의 박스를 점수순으로 훑으며 IoU 가 크거나 한쪽에 거의 포함되는 박스는 같은 객체로 보고 버린다."""
    keep = []
    for i in np.argsort(-scores):
        if scores[i] < thr: break
        dup = False
        for j in keep:
            io, ct = iou_contain(boxes[i], boxes[j])
            if io >= iou_t or ct >= contain_t: dup = True; break
        if not dup: keep.append(i)
    return keep

rows = []; why = collections.Counter()
for f in test_files:
    c, s, b = P_base[f]; ds, db = P_drone[f][1], P_drone[f][2]
    sa, sb, _ = SC[f]; sd = ds.max() if len(ds) else 0.0
    if sd >= TAU: kind, r = "drone", "drone_model"
    elif sb >= TP and sb >= sa: kind, r = "bird", "base_bird"
    elif sa >= TP: kind, r = "airplane", "base_airplane"
    elif sd >= TD: kind, r = "drone", "drone_low"
    elif sb >= TP_LO and sb >= sa: kind, r = "bird", "low_bird"
    elif sa >= TP_LO: kind, r = "airplane", "low_airplane"
    else: kind, r = "drone", "default"
    why[r] += 1
    if kind == "airplane": n = len(merge_boxes(b[c == AIR], s[c == AIR], 0.25))
    elif kind == "bird": n = len(merge_boxes(b[c == BIRD], s[c == BIRD], 0.25))
    else: n = len(merge_boxes(db, ds, TD, iou_t=0.5, contain_t=1.01))     # 드론은 포함도 병합을 쓰지 않는다
    n = max(n, 1); rows.append(dict(filename=f, airplane=n if kind == "airplane" else 0, bird=n if kind == "bird" else 0, drone=n if kind == "drone" else 0))
df = pd.DataFrame(rows)
assert len(df) == 300 and ((df[["airplane", "bird", "drone"]] > 0).sum(axis=1) == 1).all() and df.filename.is_monotonic_increasing
print("종류 판정 근거:", dict(why))
print("종류별 이미지 수:", {k: int((df[k] > 0).sum()) for k in ("airplane", "bird", "drone")}, "| 수량 합:", {k: int(df[k].sum()) for k in ("airplane", "bird", "drone")})
print("수량 분포:", {k: dict(df[k][df[k] > 0].value_counts().sort_index()) for k in ("airplane", "bird", "drone")})
df.to_csv("submission.csv", index=False)
chk = pd.read_csv("submission.csv"); assert chk.columns.tolist() == ["filename", "airplane", "bird", "drone"] and len(chk) == 300 and chk.filename.tolist() == test_files and not chk.isna().any().any()
print("제출 파일 저장 및 형식 확인:", chk.shape); chk.head()
'''
