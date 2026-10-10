# 10_weak_sum.py  ── 합 라벨 약지도: 개별 라벨 없이 "두 항목의 합"만 있을 때, 개별 분류 확률을 합성곱해 합의 확률에 직접 손실을 준다
# 언제: 라벨이 묶음 단위(두 숫자의 합, 여러 항목의 개수 합)로만 주어질 때. 모델 구조 변경이 아니라 라벨 사용 방식이므로 베이스라인에서 정한다(규칙 14.3).
# 사용 흐름 (L, R: 왼쪽·오른쪽 항목 텐서 [N, C, H, W] 를 train 평균/표준편차로 표준화한 것, ys: 합 라벨 0 ~ 2(K-1)):
#   m = fit_weak_sum(lambda: make_item_cnn(1, 10), L, R, ys, Xd, yd, epochs=2)    # 먼저 2 에폭으로 에폭당 시간을 본다(출력의 초)
#   m = fit_weak_sum(lambda: make_item_cnn(1, 10), L, R, ys, Xd, yd, epochs=20)   # Xd, yd: 개별 라벨이 조금 있으면 함께 쓴다(없으면 None)
#   q = predict_sum_proba(m, L_test, R_test); pred = q.argmax(1)                  # [N, 2K-1] 합의 확률
# 팁: 손글씨 두 숫자의 합 문제에서 개별 숫자 1,000장만 쓴 분류기 0.972 -> 합 라벨 약지도 0.986 (홀드아웃 2,000쌍).
#     시드 앙상블은 +0.05%p 로 노이즈였다. 개별 라벨 없이 합 라벨만으로 학습하는 경우의 효과는 미측정이다.
#     항목이 셋 이상이면 sum_dist_many 로 확률을 차례로 접는다. 에폭은 시간을 재고 정한다(13.7).
import os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def sum_dist(pa, pb):
    """두 항목의 클래스 확률 [N, Ka], [N, Kb] -> 합(0 ~ Ka+Kb-2)의 확률 [N, Ka+Kb-1]. 독립 가정의 합성곱."""
    ka, kb = pa.size(1), pb.size(1)
    idx = (torch.arange(ka).view(-1, 1) + torch.arange(kb).view(1, -1)).reshape(-1).to(pa.device)
    q = (pa.unsqueeze(2) * pb.unsqueeze(1)).flatten(1)
    return torch.zeros(pa.size(0), ka + kb - 1, dtype=q.dtype, device=pa.device).index_add_(1, idx, q)

def sum_dist_many(ps):
    """항목이 여러 개일 때 합의 확률(차례로 접는다)."""
    out = ps[0]
    for p in ps[1:]: out = sum_dist(out, p)
    return out

def weak_sum_loss(logit_a, logit_b, y_sum):
    """합 라벨에 대한 음의 로그우도. logit 은 항목별 분류기 출력, y_sum 은 long 텐서."""
    q = sum_dist(F.softmax(logit_a, 1), F.softmax(logit_b, 1))
    return -torch.log(q[torch.arange(len(y_sum)), y_sum] + 1e-8).mean()

def make_item_cnn(in_ch=1, n_classes=10, w=24):
    """항목 하나를 분류하는 작은 CNN(Conv-BN-ReLU x2 + MaxPool 블록 3개). 왼쪽·오른쪽에 같은 모델을 공유한다."""
    blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                     nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(blk(in_ch, w), blk(w, 2 * w), blk(2 * w, 4 * w), nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * w * 4, n_classes))

def shift_aug(x, s=2):
    """최대 s 픽셀 이동 증강(배치 전체에 같은 이동)."""
    if not s: return x
    dx, dy = np.random.randint(-s, s + 1, 2); return torch.roll(x, (int(dy), int(dx)), (2, 3))

def fit_weak_sum(make_model, L, R, ys, Xd=None, yd=None, epochs=20, seed=0, bs=128, lr=2e-3, lam=1.0, shift=2, n_threads=None, verbose=True):
    """분류기 하나를 왼쪽·오른쪽에 공유하고 합 라벨로 학습한다. 개별 라벨(Xd, yd)이 있으면 그 손실을 lam 배로 더한다."""
    torch.set_num_threads(n_threads or os.cpu_count() or 2); torch.manual_seed(seed); np.random.seed(seed); m = make_model()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(L) + bs - 1) // bs))
    ys = torch.tensor(np.array(ys), dtype=torch.long); yd = None if yd is None else torch.tensor(np.array(yd), dtype=torch.long)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(L)); t0 = time.time()
        for i in range(0, len(L), bs):
            b = perm[i:i + bs]; n = len(b); lg = m(torch.cat([shift_aug(L[b], shift), shift_aug(R[b], shift)]))
            loss_sum = weak_sum_loss(lg[:n], lg[n:], ys[b]); loss = loss_sum; loss_item = torch.zeros(())
            if Xd is not None:
                db = torch.randint(0, len(Xd), (64,)); loss_item = F.cross_entropy(m(shift_aug(Xd[db], shift)), yd[db], label_smoothing=0.05)
                loss = loss_sum + lam * loss_item
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if verbose and (ep < 2 or ep == epochs - 1 or (ep + 1) % 5 == 0):
            print(f'  epoch {ep + 1}/{epochs} {time.time() - t0:.1f}s sum-loss {loss_sum.item():.3f} item-loss {loss_item.item():.3f}', flush=True)
    return m

@torch.no_grad()
def item_proba(m, X, bs=1000):
    m.eval(); return torch.cat([F.softmax(m(X[i:i + bs]), 1) for i in range(0, len(X), bs)])

def predict_sum_proba(m, L, R):
    """왼쪽·오른쪽 항목 -> 합의 확률 [N, 2K-1]."""
    return sum_dist(item_proba(m, L), item_proba(m, R))
