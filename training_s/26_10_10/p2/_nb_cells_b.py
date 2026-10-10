CELL_F = '''# ---- 종류 판정 규칙의 근거: 정답이 있는 드론 58장에서 기본 yolov8s 가 어떻게 보는가 ----
AIR, BIRD = 4, 14
TD = 0.35        # 드론 모델 임계값(그룹 교차검증에서 0.3~0.4 가 비슷, 중간값)
TP = 0.5         # 기본 모델이 비행기/새로 '확신'하는 점수
TP_LO = 0.1      # 아무것도 확신하지 못할 때 쓰는 낮은 점수
AR_T = 2.2       # 비행기 박스의 가로/세로가 이보다 작으면 드론으로 본다(드론 정답 박스의 75% 분위수 2.14 근처)

def iou_contain(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-12), inter / (min(aa, ab) + 1e-12)

def merge_boxes(boxes, scores, thr, iou_t=0.5, contain_t=0.8):
    """점수 thr 이상의 박스를 점수순으로 훑으며, 이미 남긴 박스와 IoU 가 크거나 한쪽에 거의 포함되면 같은 객체로 보고 버린다(부분 박스 중복 방지)."""
    keep = []
    for i in np.argsort(-scores):
        if scores[i] < thr: break
        dup = False
        for j in keep:
            io, ct = iou_contain(boxes[i], boxes[j])
            if io >= iou_t or ct >= contain_t: dup = True; break
        if not dup: keep.append(i)
    return keep

def img_size(path): return Image.open(path).size

def kind_from_base(c, s, b, W, H):
    """기본 모델만 보고 종류를 정한다. 반환: (종류, 비행기 박스 가로/세로)"""
    sa = s[c == AIR].max() if (c == AIR).any() else 0.0; sb = s[c == BIRD].max() if (c == BIRD).any() else 0.0
    if sb >= TP and sb >= sa: return "bird", None
    if sa >= TP:
        i = np.where(c == AIR)[0][np.argmax(s[c == AIR])]; w, h = (b[i][2] - b[i][0]) * W, (b[i][3] - b[i][1]) * H
        ar = w / max(h, 1e-6); return ("airplane" if ar >= AR_T else "drone"), ar
    return "none", None

# 정답 드론 58장에 규칙(기본 모델 부분)을 적용해 드론을 되찾는 비율을 본다
tl_files = sorted(f for f in os.listdir(TL) if f.lower().endswith((".jpg", ".jpeg", ".png")))
base_m = YOLO("yolov8s.pt"); kinds = {}
for i in range(0, len(tl_files), 10):
    for f, r in zip(tl_files[i:i + 10], base_m.predict([os.path.join(TL, x) for x in tl_files[i:i + 10]], imgsz=640, conf=0.05, iou=0.6, device=DEVICE, verbose=False)):
        W, H = img_size(os.path.join(TL, f)); kinds[f] = kind_from_base(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy(), W, H)[0]
cnt = pd.Series(kinds).value_counts()
print("정답 드론 58장에 대한 기본 모델 판정:", dict(cnt))
print("  드론으로 되찾는 비율(기본 모델 부분만): '드론' + '확신 없음(none)' = %d/58 = %.0f%% | 비행기로 잘못 보는 비율 %d/58 | 새로 잘못 보는 비율 %d/58" % (cnt.get("drone", 0) + cnt.get("none", 0), 100 * (cnt.get("drone", 0) + cnt.get("none", 0)) / 58, cnt.get("airplane", 0), cnt.get("bird", 0)))
'''

CELL_E = '''# ---- 두 모델 결과 합치기: 종류 판정 -> 수량 -> submission.csv ----
import collections
def count_for(kind, P_b, P_d, f):
    c, s, b = P_b[f]; dc, ds, db = P_d[f]
    if kind == "airplane": n = len(merge_boxes(b[c == AIR], s[c == AIR], 0.25))
    elif kind == "bird": n = len(merge_boxes(b[c == BIRD], s[c == BIRD], 0.25))
    else: n = len(merge_boxes(db, ds, TD, iou_t=0.5, contain_t=1.01))      # 드론은 포함도 병합을 쓰지 않는다
    return max(n, 1)

rows = []; why = collections.Counter()
for f in test_files:
    c, s, b = P_base[f]; dc, ds, db = P_drone[f]; W, H = img_size(os.path.join(AO, f)); k, ar = kind_from_base(c, s, b, W, H)
    sd = ds.max() if len(ds) else 0.0; sa = s[c == AIR].max() if (c == AIR).any() else 0.0; sb = s[c == BIRD].max() if (c == BIRD).any() else 0.0
    if k == "none":
        if sd >= TD: kind, r = "drone", "drone_model"
        elif sb >= TP_LO and sb >= sa: kind, r = "bird", "low_bird"
        elif sa >= TP_LO: kind, r = "airplane", "low_airplane"
        else: kind, r = "drone", "default"
    else: kind, r = k, "base_" + k
    if kind == "drone" and sd < TD and r != "default" and k == "drone": r = "base_drone_shape"
    why[r] += 1; n = count_for(kind, P_base, P_drone, f)
    rows.append(dict(filename=f, airplane=n if kind == "airplane" else 0, bird=n if kind == "bird" else 0, drone=n if kind == "drone" else 0))
df = pd.DataFrame(rows)
assert len(df) == 300 and ((df[["airplane", "bird", "drone"]] > 0).sum(axis=1) == 1).all() and df.filename.is_monotonic_increasing
print("종류 판정 근거:", dict(why))
print("종류별 이미지 수:", {k: int((df[k] > 0).sum()) for k in ("airplane", "bird", "drone")}, "| 수량 합:", {k: int(df[k].sum()) for k in ("airplane", "bird", "drone")})
print("수량 분포:", {k: dict(df[k][df[k] > 0].value_counts().sort_index()) for k in ("airplane", "bird", "drone")})
df.to_csv("submission.csv", index=False)
chk = pd.read_csv("submission.csv"); assert chk.columns.tolist() == ["filename", "airplane", "bird", "drone"] and len(chk) == 300 and chk.filename.tolist() == test_files and not chk.isna().any().any()
print("제출 파일 저장 및 형식 확인:", chk.shape); chk.head()
'''
