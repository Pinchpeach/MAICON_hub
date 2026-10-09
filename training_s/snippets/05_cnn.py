# 05_cnn.py  ── 이미지 분류(직접 학습): 작은 CNN 베이스라인 + 시간 측정 + 증강 + 좌우반전 TTA (CPU 기준)
# 사용 흐름 (X: float32 텐서 [N, C, H, W] 를 train 평균/표준편차로 표준화한 것, y: long 텐서):
#   est = time_epochs(X_tr, y_tr, n_classes=10)             # 먼저 2 에폭의 시간을 재서 총 비용을 가늠
#   m = train_cnn(X_tr, y_tr, epochs=12, n_classes=10)      # 베이스라인 -> 에폭·증강 조정(하나씩) -> 폭·깊이 확대 순으로 확장
#   proba = predict_cnn(m, X_va)                            # 좌우반전 TTA 평균 확률
# 팁: 28x28 흑백 기준 에폭당 약 10~12초(CPU 12 스레드, 1만 장). 시드 앙상블은 효과가 확인될 때만, 해상도가 크면 에폭당 시간이 급증한다.
import os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(os.cpu_count() or 4)

def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

def make_cnn(in_ch=1, n_classes=10, w=32):
    """Conv-BN-ReLU x2 + MaxPool 블록 3개. 입력 크기는 8 이상이면 어떤 크기든 가능(AdaptiveAvgPool)."""
    return nn.Sequential(_block(in_ch, w), _block(w, 2 * w), _block(2 * w, 4 * w), nn.AdaptiveAvgPool2d(3), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(4 * w * 9, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))

def augment(xb, flip=True, shift=2):
    if flip:
        f = torch.rand(xb.size(0)) < 0.5; xb = torch.where(f.view(-1, 1, 1, 1), xb.flip(3), xb)
    if shift:
        dx, dy = np.random.randint(-shift, shift + 1, 2); xb = torch.roll(xb, (int(dy), int(dx)), (2, 3))
    return xb

def train_cnn(X, y, epochs=12, n_classes=10, w=32, seed=0, bs=128, lr=3e-3, flip=True, shift=2, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed); m = make_cnn(X.shape[1], n_classes, w)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(X) + bs - 1) // bs))
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X)); t0 = time.time()
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]; loss = F.cross_entropy(m(augment(X[idx], flip, shift)), y[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if verbose and (ep < 2 or ep == epochs - 1): print(f'  epoch {ep + 1}/{epochs} {time.time() - t0:.1f}s loss {loss.item():.3f}', flush=True)
    return m

def time_epochs(X, y, n_classes=10, w=32, n=2, planned_epochs=12):
    """2 에폭만 돌려 에폭당 시간을 재고 planned_epochs 의 예상 총 시간을 출력(학습 비용을 먼저 가늠)."""
    t0 = time.time(); train_cnn(X, y, epochs=n, n_classes=n_classes, w=w, verbose=False); per = (time.time() - t0) / n
    print(f'에폭당 {per:.1f}초 -> {planned_epochs} 에폭 약 {per * planned_epochs / 60:.1f}분'); return per

@torch.no_grad()
def predict_cnn(m, X, tta=True):
    m.eval(); out = []
    for i in range(0, len(X), 1000):
        xb = X[i:i + 1000]; p = F.softmax(m(xb), 1)
        out.append((p + F.softmax(m(xb.flip(3)), 1)) / 2 if tta else p)
    return torch.cat(out)
