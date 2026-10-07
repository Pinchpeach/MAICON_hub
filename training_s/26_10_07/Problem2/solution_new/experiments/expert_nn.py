# solution_new/experiments 폴더에서 실행 (상대 경로). route_check.py 는 BiGRU OOF 파일(p2_rnn_oof.npy)이 필요함
import re, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
torch.set_num_threads(8)
OLDSP = r'C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-root-docker11\62eee907-2c50-4e3e-a181-3dcdb4a66b71\scratchpad\\'
SP = r'C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-root-docker11\7ec5aa84-89af-4967-a0fe-dd8714961a0f\scratchpad\\'
tr = pd.read_csv(r'..\data\train.csv'); y = tr.label.values
s = np.load('v1_oof_score.npy'); main = (s >= np.quantile(s, 1 - y.mean())).astype(int)
norm = lambda t: re.sub(r'(.)\1{2,}', r'\1\1', t); txt = np.array([norm(t) for t in tr.review]); L = 140
def vocab_of(texts): c = Counter(ch for t in texts for ch in t); return {ch: i + 2 for i, ch in enumerate([k for k, n in c.items() if n >= 2])}
def encode(texts, voc):
    o = np.zeros((len(texts), L), dtype=np.int64)
    for i, t in enumerate(texts): ids = [voc.get(ch, 1) for ch in t[:L]]; o[i, :len(ids)] = ids
    return torch.tensor(o)
class GRUNet(nn.Module):
    def __init__(s_, V, use_s, E=64, H=64):
        super().__init__(); s_.use_s = use_s; s_.emb = nn.Embedding(V, E, padding_idx=0); s_.drop = nn.Dropout(0.4)
        s_.gru = nn.GRU(E, H, batch_first=True, bidirectional=True); s_.fc = nn.Linear(4 * H + (1 if use_s else 0), 1)
    def forward(s_, x, sv):
        m = (x != 0).unsqueeze(-1); h, _ = s_.gru(s_.drop(s_.emb(x))); mx = h.masked_fill(~m, -1e4).max(1).values; mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        z = torch.cat([mx, mean] + ([sv.unsqueeze(1)] if s_.use_s else []), 1); return s_.fc(s_.drop(z)).squeeze(-1)
class CNNNet(nn.Module):
    def __init__(s_, V, use_s, E=64, F_=64):
        super().__init__(); s_.use_s = use_s; s_.emb = nn.Embedding(V, E, padding_idx=0); s_.drop = nn.Dropout(0.4)
        s_.convs = nn.ModuleList([nn.Conv1d(E, F_, k, padding=k // 2) for k in (2, 3, 4, 5)]); s_.fc = nn.Linear(4 * F_ + (1 if use_s else 0), 1)
    def forward(s_, x, sv):
        e = s_.drop(s_.emb(x)).transpose(1, 2); z = torch.cat([F.relu(c(e)).max(2).values for c in s_.convs] + ([sv.unsqueeze(1)] if s_.use_s else []), 1)
        return s_.fc(s_.drop(z)).squeeze(-1)
trim = lambda xb: xb[:, :max(1, int((xb != 0).sum(1).max()))]
def fit_predict(kind, use_s, Xa, sa, ya, Xb, sb, V, epochs, seed, bs=64, lr=3e-3):
    torch.manual_seed(seed); m = (GRUNet if kind == 'gru' else CNNNet)(V, use_s); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * ((len(Xa) + bs - 1) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps); yt = torch.tensor(ya, dtype=torch.float32); sa_t = torch.tensor(sa, dtype=torch.float32)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), bs):
            idx = perm[i:i + bs]; loss = F.binary_cross_entropy_with_logits(m(trim(Xa[idx]), sa_t[idx]), yt[idx]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    m.eval(); sb_t = torch.tensor(sb, dtype=torch.float32)
    with torch.no_grad(): return torch.sigmoid(m(trim(Xb), sb_t)).numpy()
def run(gate, kind, use_s, epochs=12, seeds=2):
    idx = np.where(gate)[0]; o = np.zeros(len(idx)); lab = y[idx]
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(idx, lab):
        voc = vocab_of(txt[idx[a]]); Xa, Xb = encode(txt[idx[a]], voc), encode(txt[idx[b]], voc)
        o[b] = np.mean([fit_predict(kind, use_s, Xa, (s[idx[a]] - .5) * 4, lab[a], Xb, (s[idx[b]] - .5) * 4, len(voc) + 2, epochs, sd) for sd in range(seeds)], 0)
    return idx, o
gates = {'A. 긍정 예측 & s<0.6': (main == 1) & (s < .6), 'B. |s-0.5|<0.1': np.abs(s - .5) < .1}
res = {}
for gname, g in gates.items():
    base_acc = (main == y)[g].mean()
    print(f'\n[{gname}] n={g.sum()} | 1차 모델 정확도 {base_acc:.3f} | 라벨 1 비율 {y[g].mean():.3f} (찍기 기준 {max(y[g].mean(), 1 - y[g].mean()):.3f})', flush=True)
    for kind in ('gru', 'cnn'):
        for use_s in (False, True):
            t0 = time.time(); idx, o = run(g, kind, use_s); acc = ((o >= .5) == y[idx]).mean()
            new = main.copy(); new[idx] = (o >= .5).astype(int); res[(gname, kind, use_s)] = o
            print(f'  NN={kind:3s} 점수입력={"O" if use_s else "X"} | 의심구간 정확도 {acc:.3f} (1차 대비 {(acc - base_acc) * 100:+.1f}%p) | AUC {roc_auc_score(y[idx], o):.3f} | 전체 정확도 {(new == y).mean():.4f} ({((new == y).mean() - (main == y).mean()) * 100:+.3f}%p) | {time.time() - t0:.0f}s', flush=True)
np.save('expert_nn_res.npy', {str(k): v for k, v in res.items()}, allow_pickle=True)
