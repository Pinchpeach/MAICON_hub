# solution_new/experiments 폴더에서 실행 (상대 경로). p2_resnn.py 는 p2_clean.py 가 만든 p2_clean_store.npy 가 필요함
import time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
torch.set_num_threads(10)
SP = r'C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-root-docker11\7ec5aa84-89af-4967-a0fe-dd8714961a0f\scratchpad\\'
tr = pd.read_csv(r'..\data\train.csv'); X = tr.review.values; y = tr.label.values
store = np.load('p2_clean_store.npy', allow_pickle=True).item(); L = 140
def vocab_of(texts): c = Counter(ch for t in texts for ch in t); return {ch: i + 2 for i, ch in enumerate([k for k, n in c.items() if n >= 2])}
def encode(texts, voc):
    o = np.zeros((len(texts), L), dtype=np.int64)
    for i, t in enumerate(texts): ids = [voc.get(ch, 1) for ch in t[:L]]; o[i, :len(ids)] = ids
    return torch.tensor(o)
class ResCNN(nn.Module):
    def __init__(s_, V, E=64, F_=64):
        super().__init__(); s_.emb = nn.Embedding(V, E, padding_idx=0); s_.drop = nn.Dropout(0.4)
        s_.convs = nn.ModuleList([nn.Conv1d(E, F_, k, padding=k // 2) for k in (2, 3, 4, 5)]); s_.fc = nn.Linear(4 * F_, 1); nn.init.zeros_(s_.fc.weight); nn.init.zeros_(s_.fc.bias)
        s_.beta = nn.Parameter(torch.tensor(1.0))
    def forward(s_, x, sc):
        e = s_.drop(s_.emb(x)).transpose(1, 2); z = torch.cat([F.relu(c(e)).max(2).values for c in s_.convs], 1)
        return s_.beta * (sc - .5) * 8 + s_.fc(s_.drop(z)).squeeze(-1)           # 1차 점수 항 + NN 보정 항(0 으로 시작)
trim = lambda xb: xb[:, :max(1, int((xb != 0).sum(1).max()))]
def run(alpha, epochs=8, bs=128, lr=3e-3, seed=0):
    out = np.zeros(len(y)); t0 = time.time()
    for k in range(5):
        d = store[k]; a, b = d['a'], d['b']; voc = vocab_of(X[a]); Xa, Xb = encode(X[a], voc), encode(X[b], voc)
        sa = torch.tensor(d['s_in'], dtype=torch.float32); sb = torch.tensor(d['s_val'], dtype=torch.float32); ya = torch.tensor(y[a], dtype=torch.float32)
        h = 1 - 2 * np.abs(d['s_in'] - .5); w = torch.tensor(1 + alpha * h, dtype=torch.float32); w = w / w.mean()
        torch.manual_seed(seed); m = ResCNN(len(voc) + 2); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(a) + bs - 1) // bs))
        for ep in range(epochs):
            m.train(); perm = torch.randperm(len(a))
            for i in range(0, len(a), bs):
                idx = perm[i:i + bs]; loss = (F.binary_cross_entropy_with_logits(m(trim(Xa[idx]), sa[idx]), ya[idx], reduction='none') * w[idx]).mean()
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad(): lg = m(trim(Xb), sb).numpy()
        out[b] = 1 / (1 + np.exp(-lg)); print(f'  [alpha={alpha}] fold {k} beta={m.beta.item():.2f} {time.time() - t0:.0f}s', flush=True)
    return out
pr = y.mean(); rng = np.random.default_rng(0)
base = np.zeros(len(y), dtype=int); sv_all = np.zeros(len(y))
for k in range(5): d = store[k]; sv_all[d['b']] = d['s_val']; base[d['b']] = (d['s_val'] >= np.quantile(d['s_val'], 1 - y[d['a']].mean())).astype(int)
print('기준(NB 3종, 같은 분할) 정확도 %.4f' % (base == y).mean(), flush=True)
for alpha in (0.0, 3.0):
    p = run(alpha); pred = np.zeros(len(y), dtype=int)
    for k in range(5): b = store[k]['b']; pred[b] = (p[b] >= np.quantile(p[b], 1 - y[store[k]['a']].mean())).astype(int)
    c = (pred == y).astype(int); bb = (base == y).astype(int); dd = [c[i].mean() - bb[i].mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    conf = np.abs(sv_all - .5); band = conf < .1
    print(f'[alpha={alpha}] 잔차 NN 정확도 {c.mean():.4f} (기준 대비 {(c.mean() - bb.mean()) * 100:+.3f}%p, 95% [{np.percentile(dd, 2.5) * 100:+.2f}, {np.percentile(dd, 97.5) * 100:+.2f}]%p) | 경계선(|s-.5|<.1) 정확도 기준 {bb[band].mean():.3f} -> {c[band].mean():.3f} | 바뀐 예측 {(pred != base).sum()}개 중 맞아진 것 {((pred != base) & (pred == y)).sum()}개', flush=True)
    np.save(f'p2_resnn_p_alpha{alpha}.npy', p)
