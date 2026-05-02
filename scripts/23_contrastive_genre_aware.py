"""
Script 23 — Genre-aware contrastive learning.

Same GraphCL setup as script 22 but replaces NT-Xent with a genre-aware variant:
same-genre song pairs are excluded from the negative set so the model is not
penalised for pulling them together.

Key change — nt_xent_genre_aware():
  For each anchor i, any other song j with the SAME known genre is masked out of
  the denominator. Songs with unknown genre (label == -1) are kept as negatives
  since we can't guarantee they're the same genre.

Architecture / hyperparameters are identical to script 22 so results are
directly comparable.

Outputs (separate from script 22 so both sets are preserved):
  results/ga_contrastive_best.pt
  results/ga_contrastive_history.json
  results/ga_contrastive_training_curves.png
  results/ga_contrastive_genre_umap.png
  results/ga_contrastive_genre_acc.json

Run from project root:
    source venv/bin/activate && python scripts/23_contrastive_genre_aware.py
"""

import sys, os, json, copy, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch_geometric.data import Batch
from torch_scatter import scatter_mean

from src.model import MusicHeteroGNN
from src.dataset import get_splits

RESULTS    = 'results'
PROCESSED  = 'data/processed'
DATA_DIR   = 'data/McGill-Billboard'
GENRE_FILE = 'data/genre_labels.json'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Augmentation ──────────────────────────────────────────────────────────────

DROPPABLE = [
    ('occ',       'in_section',     'sec'),
    ('sec',       'sec_rev',        'occ'),
    ('sec',       'next_section',   'sec'),
    ('chord',     'chord_contains', 'note'),
    ('note',      'note_in_chord',  'chord'),
    ('chord',     'chord_degree',   'scale_deg'),
    ('scale_deg', 'degree_rev',     'chord'),
]


def augment(g, edge_drop=0.25, feat_mask=0.20):
    g = copy.deepcopy(g)
    for et in DROPPABLE:
        try:
            ei = g[et].edge_index
            if ei.shape[1] == 0:
                continue
            keep = torch.rand(ei.shape[1]) > edge_drop
            if keep.sum() == 0:
                keep[torch.randint(ei.shape[1], (1,))] = True
            g[et].edge_index = ei[:, keep]
        except (KeyError, AttributeError):
            pass
    mask = (torch.rand_like(g['occ'].x) > feat_mask).float()
    g['occ'].x = g['occ'].x * mask
    return g


# ── Genre-aware NT-Xent loss ──────────────────────────────────────────────────

def nt_xent_genre_aware(z1, z2, genres, temp=0.5):
    """
    z1, z2  : [N, D] L2-normalised projections of the two augmented views.
    genres  : [N] int tensor — coarse genre index per song, -1 = unknown.

    Same-genre pairs (both genre >= 0 and matching) are removed from the
    negative set so the model is not penalised for clustering them together.
    The positive pair (z1[i] ↔ z2[i]) is always kept regardless of genre.
    """
    N = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # [2N, D]
    g = torch.cat([genres, genres], dim=0)               # [2N]

    sim = torch.mm(z, z.T) / temp  # [2N, 2N]

    # Pairs to suppress in the denominator:
    # (a) self-similarity on the diagonal
    suppress = torch.eye(2 * N, dtype=torch.bool, device=z.device)

    # (b) same-genre cross-song pairs where both labels are known
    both_known  = (g.unsqueeze(0) >= 0) & (g.unsqueeze(1) >= 0)  # [2N, 2N]
    same_genre  = g.unsqueeze(0) == g.unsqueeze(1)                # [2N, 2N]
    suppress   |= (both_known & same_genre)

    # Never suppress the positive pair itself
    pos = torch.zeros(2 * N, 2 * N, dtype=torch.bool, device=z.device)
    pos[torch.arange(N), torch.arange(N, 2 * N)] = True
    pos[torch.arange(N, 2 * N), torch.arange(N)] = True
    suppress &= ~pos

    sim = sim.masked_fill(suppress, -1e9)

    labels = torch.cat([
        torch.arange(N, 2 * N),
        torch.arange(0, N),
    ], dim=0).to(z.device)

    return F.cross_entropy(sim, labels)


# ── Genre lookup helpers ───────────────────────────────────────────────────────

GENRE_TO_IDX = {
    'rock': 0, 'pop': 1, 'soul_r&b': 2, 'country': 3,
    'folk': 4, 'jazz': 5, 'blues': 6, 'disco_dance': 7,
}  # 'other' / unknown → -1


def build_sid_genre_map(genre_labels: dict) -> dict:
    """sid → int genre index (-1 if unknown/other)."""
    out = {}
    for sid, meta in genre_labels.items():
        c = meta.get('coarse', 'other') if isinstance(meta, dict) else 'other'
        out[sid] = GENRE_TO_IDX.get(c, -1)
    return out


def chunk_genre_tensor(chunk, sid_genre: dict) -> torch.Tensor:
    """Return [N] int tensor of genre indices for a list of graphs."""
    return torch.tensor(
        [sid_genre.get(getattr(g, 'song_id', ''), -1) for g in chunk],
        dtype=torch.long,
    )


# ── Model ─────────────────────────────────────────────────────────────────────

class ContrastiveGNN(nn.Module):
    def __init__(self, hidden_dim=128, proj_dim=64):
        super().__init__()
        self.encoder = MusicHeteroGNN(
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=0.3,
            use_prev_edges=False,
            use_chord_in_occ=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.hidden_dim = hidden_dim

    def _song_emb(self, batch):
        x_dict  = {k: v.to(DEVICE) for k, v in batch.x_dict.items()}
        ei_dict = {k: v.to(DEVICE) for k, v in batch.edge_index_dict.items()}
        occ_emb = self.encoder.encode_occ(x_dict, ei_dict)
        occ_bat = batch['occ'].batch.to(DEVICE)
        n_songs = int(occ_bat.max().item()) + 1
        return scatter_mean(occ_emb, occ_bat, dim=0, dim_size=n_songs)

    def forward(self, batch1, batch2, genres):
        s1 = self._song_emb(batch1)
        s2 = self._song_emb(batch2)
        return nt_xent_genre_aware(self.proj(s1), self.proj(s2),
                                   genres.to(DEVICE))

    @torch.no_grad()
    def get_embeddings(self, graphs, batch_size=32):
        self.eval()
        embs = []
        for i in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[i:i + batch_size])
            embs.append(self._song_emb(batch).cpu())
        return torch.cat(embs, dim=0).numpy()


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, all_graphs, val_graphs, sid_genre, epochs=100, lr=3e-4, batch_size=32):
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    model.to(DEVICE)
    best_val, best_ep = float('inf'), 0
    history = {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(all_graphs)
        total, n = 0.0, 0

        for i in range(0, len(all_graphs), batch_size):
            chunk = all_graphs[i:i + batch_size]
            if len(chunk) < 4:
                continue
            genres = chunk_genre_tensor(chunk, sid_genre)
            v1 = Batch.from_data_list([augment(g) for g in chunk])
            v2 = Batch.from_data_list([augment(g) for g in chunk])
            opt.zero_grad()
            loss = model(v1, v2, genres)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); n += 1

        model.eval()
        with torch.no_grad():
            val_total, val_n = 0.0, 0
            for i in range(0, len(val_graphs), batch_size):
                chunk = val_graphs[i:i + batch_size]
                if len(chunk) < 4:
                    continue
                genres = chunk_genre_tensor(chunk, sid_genre)
                v1 = Batch.from_data_list([augment(g) for g in chunk])
                v2 = Batch.from_data_list([augment(g) for g in chunk])
                val_total += model(v1, v2, genres).item(); val_n += 1

        train_loss = total / max(n, 1)
        val_loss   = val_total / max(val_n, 1)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_ep  = epoch
            torch.save(model.state_dict(), f'{RESULTS}/ga_contrastive_best.pt')

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}'
                  + (f'  ← best' if epoch == best_ep else ''), flush=True)

    print(f'\nBest val loss {best_val:.4f} at epoch {best_ep}')
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

COARSE_NAMES = list(GENRE_TO_IDX.keys())


def genre_probe(embs, all_graphs, genre_labels, train_idx, test_idx):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def collect(indices):
        X, y = [], []
        for i in indices:
            sid  = getattr(all_graphs[i], 'song_id', None)
            meta = genre_labels.get(sid)
            if meta is None:
                continue
            c = meta.get('coarse', 'other') if isinstance(meta, dict) else 'other'
            if c not in GENRE_TO_IDX:
                continue
            X.append(embs[i])
            y.append(GENRE_TO_IDX[c])
        return np.array(X), np.array(y)

    X_tr, y_tr = collect(train_idx)
    X_te, y_te = collect(test_idx)

    scaler = StandardScaler().fit(X_tr)
    clf    = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(scaler.transform(X_tr), y_tr)

    train_acc = clf.score(scaler.transform(X_tr), y_tr)
    test_acc  = clf.score(scaler.transform(X_te), y_te)

    preds   = clf.predict(scaler.transform(X_te))
    per_cls = {}
    idx_to_name = {v: k for k, v in GENRE_TO_IDX.items()}
    for idx in range(len(GENRE_TO_IDX)):
        mask = y_te == idx
        if mask.sum() > 0:
            per_cls[idx_to_name[idx]] = float((preds[mask] == y_te[mask]).mean())

    return train_acc, test_acc, per_cls


def plot_umap(embs, all_graphs, genre_labels, out_path):
    import umap.umap_ as umap_mod
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    COLORS = {
        'rock': '#e41a1c', 'pop': '#377eb8', 'soul_r&b': '#ff7f00',
        'country': '#4daf4a', 'folk': '#984ea3', 'jazz': '#a65628',
        'blues': '#f781bf', 'disco_dance': '#999999', 'other': '#dddddd',
    }

    reducer = umap_mod.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords  = reducer.fit_transform(embs)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    plotted = set()
    for i, g in enumerate(all_graphs):
        sid   = getattr(g, 'song_id', None)
        meta  = genre_labels.get(sid)
        genre = meta.get('coarse', 'other') if meta else 'other'
        if genre not in COLORS:
            genre = 'other'
        ax.scatter(coords[i, 0], coords[i, 1], c=COLORS[genre],
                   s=12, alpha=0.7, linewidths=0)
        plotted.add(genre)

    patches = [mpatches.Patch(color=COLORS[g], label=g)
               for g in list(GENRE_TO_IDX.keys()) + ['other'] if g in plotted]
    ax.legend(handles=patches, loc='upper right', framealpha=0.3,
              labelcolor='white', fontsize=9)
    ax.set_title('Genre-Aware Contrastive GNN — Song Embeddings by Genre',
                 color='white', fontsize=14)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f'  Saved {out_path}')


def plot_curves(history, out_path):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['train']) + 1)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    ax.plot(epochs, history['train'], color='#4CAF50', label='train')
    ax.plot(epochs, history['val'],   color='#FF7F0E', label='val', linestyle='--')
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('NT-Xent Loss', color='white')
    ax.set_title('Genre-Aware Contrastive GNN Training', color='white')
    ax.legend(labelcolor='white', framealpha=0.3)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f'  Saved {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path(RESULTS).mkdir(exist_ok=True)
    print(f'Device: {DEVICE}')

    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED)
    all_graphs = train_g + val_g + test_g
    train_idx  = list(range(len(train_g)))
    test_idx   = list(range(len(train_g) + len(val_g), len(all_graphs)))
    print(f'Graphs: {len(train_g)} train / {len(val_g)} val / {len(test_g)} test')

    with open(GENRE_FILE) as f:
        genre_labels = json.load(f)

    sid_genre = build_sid_genre_map(genre_labels)
    labeled   = sum(1 for v in sid_genre.values() if v >= 0)
    print(f'Genre labels: {labeled}/{len(sid_genre)} songs have known coarse genre')

    model = ContrastiveGNN(hidden_dim=128, proj_dim=64)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    print('\n=== Genre-Aware Contrastive Training ===')
    history = train(model, train_g, val_g, sid_genre, epochs=100, lr=3e-4, batch_size=32)

    with open(f'{RESULTS}/ga_contrastive_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_curves(history, f'{RESULTS}/ga_contrastive_training_curves.png')

    model.load_state_dict(torch.load(f'{RESULTS}/ga_contrastive_best.pt', map_location=DEVICE))
    model.to(DEVICE)

    print('\nExtracting embeddings...')
    embs = model.get_embeddings(all_graphs, batch_size=32)
    print(f'  Embeddings shape: {embs.shape}')

    print('\n=== Genre Linear Probe ===')
    train_acc, test_acc, per_cls = genre_probe(embs, all_graphs, genre_labels, train_idx, test_idx)
    print(f'  Train acc: {train_acc:.3f}')
    print(f'  Test  acc: {test_acc:.3f}')
    print('  Per-class test accuracy:')
    for g, a in sorted(per_cls.items(), key=lambda x: -x[1]):
        print(f'    {g:15s}: {a:.3f}')

    results = {
        'train_acc': train_acc,
        'test_acc':  test_acc,
        'per_class': per_cls,
        'baselines': {
            'frozen_probe':      0.239,
            'multitask_het_gnn': 0.358,
            'hgt_hierarchical':  0.507,
            'contrastive_v1':    0.538,
        },
    }
    with open(f'{RESULTS}/ga_contrastive_genre_acc.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nGenerating UMAP...')
    plot_umap(embs, all_graphs, genre_labels, f'{RESULTS}/ga_contrastive_genre_umap.png')

    print('\nDone.')
    print(f'  Genre probe test accuracy: {test_acc:.1%}')
    print(f'  Baselines: frozen 23.9% | multitask 35.8% | HGT hier. 50.7% | contrastive v1 53.8%')


if __name__ == '__main__':
    main()
