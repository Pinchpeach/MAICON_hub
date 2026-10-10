"""문제 2·4 최종 노트북 생성(외부 .py 의존 없음, CPU 기준). 사용: python _exp/build_nb.py"""
import nbformat as nbf

LIB = r'''
GROUPS = {"person": [0], "vehicle": [2, 5, 7]}      # COCO class id: person=0, car=2, bus=5, truck=7 (자전거·오토바이 제외)

def _collect(res, x0=0.0, y0=0.0, sx=1.0, sy=1.0, flip=False):
    """예측 결과를 전체 이미지 기준 정규화 xyxy 로 변환."""
    b = res.boxes.xyxyn.cpu().numpy().copy(); s = res.boxes.conf.cpu().numpy(); c = res.boxes.cls.cpu().numpy().astype(int)
    if flip: b[:, [0, 2]] = 1 - b[:, [2, 0]]
    b[:, [0, 2]] = x0 + b[:, [0, 2]] * sx; b[:, [1, 3]] = y0 + b[:, [1, 3]] * sy
    return b, s, c

def raw_views(path, keys):
    """한 이미지를 여러 관점(해상도, 좌우 반전, 2x2 타일)으로 추론해 원시 박스를 모은다. 깨진 이미지는 빈 결과."""
    try:
        arr = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]      # BGR
    except Exception as e:
        print("이미지 읽기 실패 → 0으로 처리:", path, e); z = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)); return {k: z for k in keys}
    H, W = arr.shape[:2]; kw = dict(conf=0.03, iou=0.7, device=DEVICE, verbose=False, max_det=300, workers=0); out = {}
    for k in keys:
        if k.startswith("full"): out[k] = _collect(model.predict(arr, imgsz=int(k[4:]), **kw)[0])
        elif k.startswith("flip"): out[k] = _collect(model.predict(arr[:, ::-1].copy(), imgsz=int(k[4:]), **kw)[0], flip=True)
        elif k.startswith("tile"):                                       # 2x2 타일(겹침 25%), 작은 객체용
            tw, th = int(W * 0.625), int(H * 0.625); parts = []
            for x in (0, W - tw):
                for y in (0, H - th):
                    parts.append(_collect(model.predict(arr[y:y + th, x:x + tw].copy(), imgsz=int(k[4:]), **kw)[0], x / W, y / H, tw / W, th / H))
            out[k] = tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
    return out

def _iou_inter(a, B):
    iw = np.maximum(0, np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0])); ih = np.maximum(0, np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]))
    inter = iw * ih; aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa + ab - inter + 1e-12), inter / (np.minimum(aa, ab) + 1e-12)

def merge(views, keys, group, nms_iou=0.5, contain=0.8, vote=1, min_ratio=0.0):
    """관점들의 박스를 합쳐 점수순으로 군집화한다(묶음 안에서는 클래스 구분 없음: car/bus/truck 이중 계수 방지).
    IoU >= nms_iou 이거나 작은 박스가 contain 이상 포함되면 같은 객체로 본다. 군집 점수 = 관점별 최고 conf 합 / 관점 수.
    vote = 그 객체를 본 최소 관점 수. min_ratio > 0 이면 포함 관계라도 면적 비율(작은/큰)이 그보다 작을 때는 다른 객체로 둔다
    (큰 트럭 박스 안에 보이는 뒤쪽의 작은 차를 따로 세기 위함)."""
    B, S, V = [], [], []
    for vi, k in enumerate(keys):
        b, s, c = views[k]; m = np.isin(c, GROUPS[group]); B.append(b[m]); S.append(s[m]); V.append(np.full(int(m.sum()), vi))
    B = np.concatenate(B); S = np.concatenate(S); V = np.concatenate(V); A = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]); reps, members = [], []
    for i in np.argsort(-S):
        if reps:
            io, ct = _iou_inter(B[i], B[reps]); ratio = np.minimum(A[i], A[reps]) / (np.maximum(A[i], A[reps]) + 1e-12)
            j = np.where((io >= nms_iou) | ((ct >= contain) & (ratio >= min_ratio)))[0]
            if len(j): members[j[0]].append(i); continue
        reps.append(i); members.append([i])
    scores = []
    for mem in members:
        best = {}
        for i in mem: best[V[i]] = max(best.get(V[i], 0), S[i])
        if len(best) >= vote: scores.append(sum(best.values()) / len(keys))
    return np.array(scores)
'''

def build(p, title, targets, views, cfg, low_code, boost_code, exp_md, out):
    nb = nbf.v4.new_notebook(); C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n"))); code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
    md(f"""
# {title}

- 목표: test 이미지 30장의 수량 예측. 점수 = 100 / (1 + Count MAE). 제출 `{','.join(['filename'] + [t + '_count' for t in targets])}`
- 주어진 것: 이미지 30장 + **수량 라벨만**(박스 라벨 없음), COCO 사전학습 `yolov8n.pt`
- 접근: 박스 라벨이 없어 검출기를 새로 학습할 수 없으므로, **사전학습 yolov8n 추론을 여러 관점(해상도·좌우 반전·타일)으로 모아 융합**하고, 공개 학습 30장의 수량 라벨로 임계값을 정한다.
- 환경: CPU 기준(`device='cpu'`), 추가 설치 없음. `data/image_credits.csv`(COCO 이미지 ID)는 정답 조회가 가능한 정보라 사용하지 않는다.
""")
    code("""
import time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
warnings.filterwarnings('ignore'); T0 = time.time()
DEVICE = 'cpu'                                   # 대회 환경 기준
DATA = next(p for p in [Path('data'), Path('/workspace/%s/data')] if (p / 'sample_submission.csv').exists())
WEIGHTS = next(p for p in [Path('yolov8n.pt'), DATA.parent / 'yolov8n.pt'] if p.exists())
sample = pd.read_csv(DATA / 'sample_submission.csv'); train_labels = pd.read_csv(DATA / 'train_labels.csv')
model = YOLO(str(WEIGHTS))
print('train_labels', train_labels.shape, '| sample', sample.shape, '| weights', WEIGHTS)
""".replace("%s", p))
    md("## 1. 데이터 분석")
    code(f"""
TARGETS = {targets!r}
for t in TARGETS:
    y = train_labels[t + '_count']
    print(f'{{t}}_count: 평균 {{y.mean():.2f}}, 표준편차 {{y.std(ddof=0):.2f}}, 범위 {{y.min()}}~{{y.max()}}, 분포 {{dict(y.value_counts().sort_index())}}')
sizes = pd.Series([Image.open(DATA / 'train_images' / f).size for f in train_labels.filename]).value_counts().head(4)
tsizes = pd.Series([Image.open(DATA / 'test_images' / f).size for f in sample.filename]).value_counts().head(4)
print('train 이미지 크기 상위:', dict(sizes)); print('test  이미지 크기 상위:', dict(tsizes))
assert set(train_labels.filename) == {{p.name for p in (DATA / 'train_images').iterdir()}} and set(sample.filename) == {{p.name for p in (DATA / 'test_images').iterdir()}}
print('박스 라벨 없음 → 수량 라벨은 임계값 선택과 검증에만 사용')
""")
    md("## 2. 추론·융합 함수\n이미지 하나를 여러 관점으로 추론해 박스를 모으고, 같은 객체끼리 묶어 점수를 합친 뒤 임계값 이상인 묶음 수를 센다.")
    code(LIB)
    md("## 3. 공개 학습 30장으로 설정 검증\n템플릿 기본값(imgsz 416, conf 0.25)과 비교하고, 임계값은 이웃 임계값까지 평균한 MAE(고원)가 가장 낮은 지점으로 정한다. LOO는 29장으로 임계값을 고르고 남은 1장을 평가한 값이다.")
    code(f"""
VIEWS = {views!r}                                # 추론 관점
CFG = {cfg!r}   # 대상: (사용 관점, NMS IoU, 포함도, 최소 관점 수, 포함 병합의 최소 면적 비율)
LOW = {low_code}   # 예측 하한(정답 분포에서 온 제약, 아래 설명)
BOOST = {boost_code}   # 센 박스의 평균 확신도가 이 값보다 낮으면 +1 (어려운 장면의 과소 계수 보정, None 이면 미적용)

def final_count(scores, thr, low, boost):
    s = scores[scores >= thr]; c = max(len(s), low)
    if boost is not None and (s.mean() if len(s) else 0.0) < boost: c += 1
    return int(c)
t1 = time.time()
train_views = {{f: raw_views(DATA / 'train_images' / f, VIEWS) for f in train_labels.filename}}
print('train 30장 추론 %.0f초 (관점 %d개)' % (time.time() - t1, len(VIEWS)))
TH = np.round(np.arange(0.05, 0.61, 0.025), 3); THR = {{}}; rows = []
for t in TARGETS:
    y = train_labels[t + '_count'].values; keys, nms, ct, vote, mr = CFG[t]
    # 템플릿 기준: 416 한 장, conf 0.25
    base = []
    for f in train_labels.filename:
        r = model.predict(str(DATA / 'train_images' / f), device=DEVICE, imgsz=416, conf=0.25, workers=0, verbose=False)[0]
        base.append(int(np.isin(r.boxes.cls.cpu().numpy().astype(int), GROUPS[t]).sum()))
    S = [merge(train_views[f], keys, t, nms, ct, vote, mr) for f in train_labels.filename]
    E0 = np.array([np.abs(np.array([(s >= th).sum() for s in S]) - y) for th in TH])            # 하한 적용 전
    E = np.array([np.abs(np.maximum(np.array([(s >= th).sum() for s in S]), LOW[t]) - y) for th in TH])
    plateau = np.array([E[max(0, k - 1):k + 2].mean() for k in range(len(TH))]); k = int(np.argmin(plateau)); THR[t] = float(TH[k])
    loo = np.mean([E[int(np.argmin(np.delete(E, i, 1).mean(1))), i] for i in range(len(y))])
    pred0 = np.maximum(np.array([(s >= THR[t]).sum() for s in S]), LOW[t])
    pred = np.array([final_count(s, THR[t], LOW[t], BOOST[t]) for s in S])
    if BOOST[t] is not None:
        ms = np.array([(s[s >= THR[t]].mean() if (s >= THR[t]).any() else 0.0) for s in S])
        print(t, '평균 확신도 기준별 MAE(+1 보정):', ' '.join(f'{{m:.2f}}:{{np.abs(pred0 + (ms < m) - y).mean():.2f}}' for m in np.arange(0.40, 0.76, 0.05)), '| 보정 대상 train %d/30장' % int((ms < BOOST[t]).sum()))
    rows.append({{'대상': t, '템플릿 MAE(416, 0.25)': np.abs(np.array(base) - y).mean(), '최종 MAE': np.abs(pred - y).mean(), '보정 전 MAE': E[k].mean(), '하한 적용 전 MAE': E0[k].mean(), '하한': LOW[t], '임계값': THR[t], 'LOO MAE': loo,
                 '완전 일치': (pred == y).mean(), '편향': (pred - y).mean(), '평균값만 예측 MAE': np.abs(y - np.round(y.mean())).mean()}})
    print(t, '임계값별 MAE:', ' '.join(f'{{th}}:{{m:.2f}}' for th, m in list(zip(TH, E.mean(1)))[1:13]))
val_table = pd.DataFrame(rows).set_index('대상').round(3); print(val_table.to_string())
mae_mean = val_table['최종 MAE'].mean(); loo_mean = val_table['LOO MAE'].mean()
print('(LOO MAE 는 확신도 보정 전, 임계값만 고른 값)')
print('공개 학습 기준 점수 추정: 100/(1+MAE) = %.1f (최종, in-sample), %.1f (보정 전 LOO) | 템플릿 %.1f' % (100 / (1 + mae_mean), 100 / (1 + loo_mean), 100 / (1 + val_table['템플릿 MAE(416, 0.25)'].mean())))
""")
    md(exp_md)
    md("## 5. test 추론, 분포 점검, 제출 저장")
    code("""
t1 = time.time()
test_views = {f: raw_views(DATA / 'test_images' / f, VIEWS) for f in sample.filename}
print('test 30장 추론 %.0f초' % (time.time() - t1))
sub = pd.DataFrame({'filename': sample.filename})
for t in TARGETS:
    keys, nms, ct, vote, mr = CFG[t]
    St = [merge(test_views[f], keys, t, nms, ct, vote, mr) for f in sample.filename]; raw_cnt = np.array([int((s >= THR[t]).sum()) for s in St])
    sub[t + '_count'] = [final_count(s, THR[t], LOW[t], BOOST[t]) for s in St]
    print(f'{t}: 하한 {LOW[t]} 로 올린 test 이미지 {int((raw_cnt < LOW[t]).sum())}장 | 확신도 보정(+1) test 이미지 {int((sub[t + "_count"].values - np.maximum(raw_cnt, LOW[t])).sum())}장')
    trp = np.array([final_count(merge(train_views[f], keys, t, nms, ct, vote, mr), THR[t], LOW[t], BOOST[t]) for f in train_labels.filename])
    print(f"{t}: 평균 수량 train 정답 {train_labels[t + '_count'].mean():.2f} / train 예측 {trp.mean():.2f} / test 예측 {sub[t + '_count'].mean():.2f} | test 분포 {dict(sub[t + '_count'].value_counts().sort_index())}")
sub = sub[list(sample.columns)].astype({c: int for c in sample.columns[1:]})
sub.to_csv('submission.csv', index=False)
check = pd.read_csv('submission.csv')
assert check.columns.tolist() == sample.columns.tolist()
assert check.iloc[:, 0].tolist() == sample.iloc[:, 0].tolist()
assert check.iloc[:, 0].is_unique and not check.isna().any().any()
assert (check.iloc[:, 1:] >= 0).all().all() and all(d.kind == 'i' for d in check.dtypes[1:])
print('제출 형식 확인:', check.shape, '| 총 소요 %.0f초' % (time.time() - T0))
""")
    nb["cells"] = C; nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    nbf.write(nb, out); print("built", out, len(C))

EXP2 = """
## 4. 학습 설계와 실험 기록 (로컬 GPU로 실험, 채택 여부만 반영)

표의 수치는 GPU 실험값이다. 위 3번 셀의 출력은 이 노트북을 CPU로 실행한 값이라 조금 다를 수 있다(템플릿 MAE처럼 저해상도·경계 conf 에서 차이가 크다).

| 시도 | 내용 | 결과(공개 학습 30장) | 판정 |
|---|---|---|---|
| 템플릿 | imgsz 416, conf 0.25 | MAE 2.47 | 기준 |
| 단일 해상도 | 640 / 960 / 1280 + conf 탐색 | 1.6~1.2 | 해상도가 클수록 작은 사람을 더 찾음 |
| **다중 관점 융합** | 960 + 1280 + 2x2 타일, 포함도 0.7, 2개 관점 이상 | **0.97 (LOO 0.97)** | **채택** |
| 관점 추가 | 1600, 3x3 타일, 타일 반전 등 | 0.93~1.17 (LOO) | 차이 노이즈, 단순한 쪽 유지 |
| 보정 회귀 | 임계값별 개수·점수합 → 수량 회귀(ridge, LOO) | 1.23~2.07 | 기각 |
| 의사 박스 미세조정 | 수량만큼 상위 박스를 라벨로 yolov8n 20에폭(5겹 OOF) | 1.17~1.77 | 기각(30장으로는 사전학습 모델을 해침) |

| 밝기 정규화(자동 대비·감마·CLAHE·평활화·선명화) | 원본 대체/병행, 밝기별 임계값 | LOO 1.00~1.70 | 기각(오류는 밝기와 무관, 상관 −0.09) |
| 임계값 아래 후보 2차 분류 | 크기별 임계값, 확대 재검출, 개수 손실 분류기 | LOO 1.03~1.33 | 기각 |
| **예측 하한 4** | 공개 학습 정답이 전부 4~13명(4명이 7장)이라 4 미만 예측을 4로 올림 | **0.97 → 0.77 (LOO 0.77)** | **채택(가정 있음)** |

**하한 4의 가정**: 평가 이미지도 같은 기준(4명 이상)으로 골랐다고 본 것이다. 평가에 4명 미만 이미지가 있으면 그 이미지에서는 손해다. 근거는 공개 학습 30장의 분포뿐이다.

| **평균 확신도 < 0.55 이면 +1** | 센 박스들의 평균 확신도가 낮은(어려운) 장면은 덜 세는 경향(오차와 순위 상관 −0.50) | 0.77 → 0.63 (LOO 0.63) | 미채택(v4로 잠시 메인이었다가 사용자 결정으로 v3 복귀) |

**확신도 보정을 뺀 이유**: 평균 확신도 분포가 평가 쪽이 더 낮아(중앙값 train 0.65 / test 0.57) 평가 30장 중 13장에 +1 이 붙는다. 평가가 학습과 같은 난이도라면 과다 계수가 된다. 해당 버전은 `versions/v4_lowconf_plus1_NOT_MAIN.csv` 로 보관.

| 축소·확대 변형으로 학습 세트 늘리기 | 늘린 세트로 임계값 재선택 / 의사 박스 20에폭 미세조정 | 0.77 (변화 없음) / 1.40~1.53 | 기각 |
| LLM 검토 박스(정답 수량과 교차검증한 27장) 미세조정 | yolov8n 20에폭, 5겹 OOF | 1.03~1.13 (0.5 축소본에서는 1.13~1.17 로 사전학습 1.43 보다 좋음) | 미채택(원본에서 사전학습 0.77 을 못 넘음) |

남은 오류는 수평선 근처의 아주 작은 사람, 손·머리로 가려진 사람 누락이 대부분이다.
"""
EXP4 = """
## 4. 학습 설계와 실험 기록 (로컬 GPU로 실험, 채택 여부만 반영)

표의 수치는 GPU 실험값이다. 위 3번 셀의 출력은 이 노트북을 CPU로 실행한 값이라 조금 다를 수 있다(템플릿 MAE처럼 저해상도·경계 conf 에서 차이가 크다).

| 시도 | 내용 | 결과(공개 학습 30장, person / vehicle MAE) | 판정 |
|---|---|---|---|
| 템플릿 | imgsz 416, conf 0.25 | 1.00 / 0.70 | 기준 |
| 단일 해상도 | 640 / 960 / 1280 + conf 탐색 | 0.60~0.67 / 0.53~0.63 | |
| **다중 관점 융합** | person: 640·960·1280 + 좌우 반전, vehicle: 640·960·1280(2개 관점 이상, car/bus/truck 묶어 중복 제거) | **0.33 / 0.50 (LOO 0.33 / 0.53)** | **채택** |
| 관점 추가 | 타일, 1600 등 | 0.33 / 0.57 | 차이 없음, 단순한 쪽 유지 |
| 보정 회귀 | 임계값별 개수 → 수량 회귀(LOO) | 0.33~0.77 / 0.50~1.03 | 기각 |
| 의사 박스 미세조정 | yolov8n 20에폭(5겹 OOF) | 0.43 / 0.53 | 기각(사전학습보다 나쁘거나 같음) |

| 밝기 정규화, 임계값 아래 후보 2차 검증(확대 재검출) | | person 0.30~0.47 / vehicle 0.40~0.73 | 기각·미채택(개선이 3장뿐이고 눈으로 보니 1개만 진짜 차량) |
| **포함 병합에 면적 비율 조건(vehicle)** | 큰 차량 박스 안의 작은 차량을 따로 셈(면적 비율 0.1 미만이면 병합 안 함) | **vehicle 0.50 → 0.43 (LOO 0.47)** | **채택**(`S12_2_0010`: 픽업트럭 뒤의 차 2대를 되찾음, 눈으로 확인) |

차량의 남은 오류는 비 맞은 유리창 너머의 차, 차량 내부에서 찍은 사진, 멀리 있는 작은 차처럼 사전학습 모델이 못 보는 경우다.
"""
build("problem02", "MAICON 모의고사 · 2번 사진 속 인원 집계", ["person"], ["full960", "full1280", "tile960"],
      {"person": (["full960", "full1280", "tile960"], 0.5, 0.7, 2, 0.0)}, "{'person': int(train_labels.person_count.min())}", "{'person': None}", EXP2, "problem02/work.ipynb")
build("problem04", "MAICON 모의고사 · 4번 사진 속 인원·차량 집계", ["person", "vehicle"], ["full640", "flip640", "full960", "flip960", "full1280", "flip1280"],
      {"person": (["full640", "flip640", "full960", "flip960", "full1280", "flip1280"], 0.5, 0.8, 1, 0.0), "vehicle": (["full640", "full960", "full1280"], 0.5, 0.7, 2, 0.1)}, "{'person': 0, 'vehicle': 0}", "{'person': None, 'vehicle': None}", EXP4, "problem04/work.ipynb")
