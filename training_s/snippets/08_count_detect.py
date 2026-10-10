# 08_count_detect.py  ── 객체 탐지 후 수량 세기: 사전학습 YOLO 다중 관점 추론 + 묶음 병합 + 고원 임계값 (학습 없음)
# 언제: 세는 대상이 사전학습(COCO 80종) 클래스일 때. 수량 라벨만 있는 수십 장이면 미세조정하지 않는다(규칙 14.3).
# 사용 흐름:
#   model = load_yolo('yolov8n.pt')                               # ultralytics·가중치는 서버에 없다 -> 설치·다운로드는 허락(13.5) 후. CPU 스레드도 여기서 맞춘다
#   time_views(model, paths[:3], ALL_KEYS)                        # 먼저 1장당 시간을 재서 train+test 전체 비용을 가늠
#   tv = {f: raw_views(model, img_dir / f, ALL_KEYS) for f in train_files}     # 관점별 원시 박스(한 번만 추론해 두고 재사용)
#   compare_viewsets(tv, train_files, y_count, COCO['person'])    # 장면에 맞는 관점 묶음을 학습 수량 라벨로 고른다
#   cfg = VIEWSETS['few_large']; S = [merge(tv[f], cfg['keys'], COCO['person'], **cfg['merge']) for f in train_files]
#   thr, info = pick_threshold(S, y_count, metric='mae')         # 이웃 평균(고원)이 가장 좋은 임계값 + LOO 값
#   test_counts = [count(merge(raw_views(model, p, cfg['keys']), cfg['keys'], COCO['person'], **cfg['merge']), thr) for p in test_paths]
# 팁: 관점 묶음은 장면마다 다르다. 작고 많은 객체 = small_many, 적고 큰 객체 = few_large, 차량 = vehicle(좌우 반전 효과 없음).
#     두 인원 문제의 묶음을 서로 바꾸면 MAE 가 0.93 -> 1.47, 0.33 -> 0.60 으로 나빠졌다(CPU 재측정). 관점·병합·임계값을 한꺼번에 고르면 과적합한다.
#     min_ratio(면적 비율)는 포함도 병합이 따로 보이는 작은 객체를 삼키는 것을 눈으로 확인했을 때만 쓴다(선택 사항).
import os, time
import numpy as np

COCO = {'person': [0], 'vehicle': [2, 5, 7]}                      # car=2, bus=5, truck=7 (자전거 1·오토바이 3 은 문제 정의를 확인하고 넣는다)
VIEWSETS = {
    'small_many': {'keys': ['full960', 'full1280', 'tile960'], 'merge': dict(nms_iou=0.5, contain=0.7, vote=2)},
    'few_large': {'keys': ['full640', 'flip640', 'full960', 'flip960', 'full1280', 'flip1280'], 'merge': dict(nms_iou=0.5, contain=0.8, vote=1)},
    'vehicle': {'keys': ['full640', 'full960', 'full1280'], 'merge': dict(nms_iou=0.5, contain=0.7, vote=2)},
}
ALL_KEYS = ['full640', 'flip640', 'full960', 'flip960', 'full1280', 'flip1280', 'tile960']

def load_yolo(weights='yolov8n.pt', n_threads=None, device='cpu'):
    """이미 받아 둔 가중치만 연다(없으면 자동 다운로드하지 않고 멈춘다).
    ultralytics 는 첫 CPU 호출에서 torch 스레드를 min(8, 코어 수 - 1)로 바꾼다. 2코어 서버에서는 1스레드가 되므로 여기서 되돌린다."""
    if not os.path.exists(str(weights)):
        raise FileNotFoundError(f'{weights} 없음: 파일명·출처·용량을 밝히고 허락을 받은 뒤 내려받는다')
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError('ultralytics 미설치: 첫 셀에서 !pip install (허락·서버 호환 확인 후)')
    import torch
    model = YOLO(str(weights)); model.predict(np.zeros((64, 64, 3), np.uint8), imgsz=64, device=device, verbose=False)   # 준비 호출(이때 스레드가 바뀐다)
    torch.set_num_threads(n_threads or os.cpu_count() or 1); print('torch 스레드:', torch.get_num_threads())
    return model

def _collect(res, x0=0.0, y0=0.0, sx=1.0, sy=1.0, flip=False):
    """예측 결과를 전체 이미지 기준 정규화 xyxy 로 바꾼다."""
    b = res.boxes.xyxyn.cpu().numpy().copy(); s = res.boxes.conf.cpu().numpy(); c = res.boxes.cls.cpu().numpy().astype(int)
    if flip: b[:, [0, 2]] = 1 - b[:, [2, 0]]
    b[:, [0, 2]] = x0 + b[:, [0, 2]] * sx; b[:, [1, 3]] = y0 + b[:, [1, 3]] * sy
    return b, s, c

def raw_views(model, path, keys, device='cpu', conf=0.03):
    """한 이미지를 여러 관점으로 추론해 원시 박스를 모은다. key = full<크기> | flip<크기>(좌우 반전) | tile<크기>(2x2 타일, 겹침 25%).
    깨진 이미지는 빈 결과(수량 0)로 처리한다."""
    from PIL import Image
    try:
        arr = np.asarray(Image.open(path).convert('RGB'))[:, :, ::-1]          # BGR
    except Exception as e:
        print('이미지 읽기 실패 -> 0 으로 처리:', path, e); z = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)); return {k: z for k in keys}
    H, W = arr.shape[:2]; kw = dict(conf=conf, iou=0.7, device=device, verbose=False, max_det=300, workers=0); out = {}
    for k in keys:
        if k.startswith('full'): out[k] = _collect(model.predict(arr, imgsz=int(k[4:]), **kw)[0])
        elif k.startswith('flip'): out[k] = _collect(model.predict(arr[:, ::-1].copy(), imgsz=int(k[4:]), **kw)[0], flip=True)
        elif k.startswith('tile'):
            tw, th = int(W * 0.625), int(H * 0.625); parts = []
            for x in (0, W - tw):
                for y in (0, H - th):
                    parts.append(_collect(model.predict(arr[y:y + th, x:x + tw].copy(), imgsz=int(k[4:]), **kw)[0], x / W, y / H, tw / W, th / H))
            out[k] = tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
        else: raise ValueError(k)
    return out

def time_views(model, paths, keys, device='cpu'):
    """몇 장으로 1장당 추론 시간을 잰다(첫 장은 준비 시간이 섞이므로 따로 한 번 돌린다)."""
    raw_views(model, paths[0], keys[:1], device); t0 = time.time()
    for p in paths: raw_views(model, p, keys, device)
    sec = (time.time() - t0) / len(paths); print(f'1장당 {sec:.2f}초 (관점 {len(keys)}개) -> 100장 약 {sec * 100 / 60:.1f}분'); return sec

def _iou_inter(a, B):
    iw = np.maximum(0, np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0])); ih = np.maximum(0, np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]))
    inter = iw * ih; aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa + ab - inter + 1e-12), inter / (np.minimum(aa, ab) + 1e-12)

def merge(views, keys, class_ids, nms_iou=0.5, contain=0.8, vote=1, min_ratio=0.0):
    """관점들의 박스를 합쳐 점수순으로 묶는다. class_ids 안에서는 클래스를 구분하지 않는다(car/bus/truck 이중 계수 방지).
    IoU >= nms_iou 이거나 작은 박스가 contain 이상 포함되면 같은 객체. 묶음 점수 = 관점별 최고 conf 의 합 / 관점 수.
    vote = 그 객체를 본 최소 관점 수. min_ratio > 0 이면 포함 관계라도 면적 비율(작은/큰)이 그보다 작을 때는 다른 객체로 둔다.
    서로 다른 묶음(사람과 차량)은 따로 호출한다(묶음 사이에는 중복 제거를 하지 않는다). 반환: 객체별 점수 배열."""
    B, S, V = [], [], []
    for vi, k in enumerate(keys):
        b, s, c = views[k]; m = np.isin(c, class_ids); B.append(b[m]); S.append(s[m]); V.append(np.full(int(m.sum()), vi))
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

def count(scores, thr, low=0):
    """임계값 이상인 객체 수. low 는 예측 하한(정답 분포에서 온 제약일 때만, 가정과 위험을 기록하고 쓴다)."""
    return int(max(int((np.asarray(scores) >= thr).sum()), low))

def pick_threshold(S, y, grid=None, metric='mae', low=0):
    """이웃 값 평균(고원)이 가장 좋은 임계값을 고른다. metric='mae'(낮을수록 좋음) 또는 'exact'(완전 일치율, 높을수록 좋음).
    반환: (임계값, {'value': 그 임계값의 지표, 'loo': 임계값만 폴드 밖에서 고른 LOO 지표, 'bias': 예측-정답 평균, 'table': 임계값별 지표})."""
    y = np.asarray(y); grid = np.round(np.arange(0.05, 0.61, 0.025), 3) if grid is None else np.asarray(grid)
    P = np.array([[count(s, th, low) for s in S] for th in grid])                 # (임계값, 이미지)
    L = np.abs(P - y) if metric == 'mae' else (P != y).astype(float)              # 손실 행렬(둘 다 낮을수록 좋게 통일)
    plateau = np.array([L[max(0, k - 1):k + 2].mean() for k in range(len(grid))]); k = int(np.argmin(plateau))
    loo = float(np.mean([L[int(np.argmin(np.delete(L, i, 1).mean(1))), i] for i in range(len(y))]))
    conv = (lambda v: v) if metric == 'mae' else (lambda v: 1 - v)
    return float(grid[k]), {'value': float(conv(L[k].mean())), 'loo': float(conv(loo)), 'bias': float((P[k] - y).mean()),
                            'table': dict(zip(grid.tolist(), np.round(conv(L.mean(1)), 3).tolist()))}

def compare_viewsets(views, files, y, class_ids, sets=None, metric='mae', low=0, min_ratio=0.0):
    """미리 추론해 둔 views(ALL_KEYS 로 만든 것)에서 관점 묶음별 지표를 비교한다. 차이가 표준오차 안이면 추론이 적은 쪽을 쓴다."""
    sets = sets or VIEWSETS; y = np.asarray(y); out = {}
    for name, cfg in sets.items():
        S = [merge(views[f], cfg['keys'], class_ids, min_ratio=min_ratio, **cfg['merge']) for f in files]
        thr, info = pick_threshold(S, y, metric=metric, low=low); out[name] = (thr, info); nk = len(cfg['keys'])
        print(f'  {name:11s} 관점 {nk}개 | 임계값 {thr:.3f} | {metric} {info["value"]:.3f} | LOO {info["loo"]:.3f} | 편향 {info["bias"]:+.2f}')
    print(f'  (이미지 {len(y)}장: 1장 차이가 {1 / len(y):.3f} 이므로 그보다 작은 차이는 구분되지 않는다)')
    return out
