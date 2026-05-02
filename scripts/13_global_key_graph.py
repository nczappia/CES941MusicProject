"""
Script 13 — Global cross-song heterogeneous graph with key normalization

Two simultaneous innovations:
1. Key normalization: all chords transposed to C major/A minor via
   Krumhansl-Schmuckler key detection. I→IV→V always maps to the same
   chord IDs regardless of what key the song is in.

2. Global graph: instead of 890 separate per-song graphs, one large
   HeteroData with shared global chord nodes (145 types). Songs that
   use the same chord types are implicitly connected — the GNN can
   propagate information across songs through shared chord nodes.
   New 'song' node type aggregates from its occ nodes.

Model: GlobalMusicGNN — causal, note edges, song nodes, genre head.

Outputs:
    results/global_gnn_best.pt
    results/global_gnn_history.json
    results/global_gnn_training_curves.png
    results/global_gnn_genre_umap.png
    results/global_gnn_genre_acc.json

Run from project root:
    source venv/bin/activate && python scripts/13_global_key_graph.py
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from src.parse   import load_all_songs
from src.graph   import build_global_heterograph
from src.model   import GlobalMusicGNN, NUM_CLASSES
from src.dataset import get_splits   # for song-level splits

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'
GENRE_PATH    = 'data/genre_labels.json'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

GENRE_WEIGHT    = 0.5
EPOCHS          = 60
LR              = 1e-3
MIN_GENRE_COUNT = 10


# ── Genre helpers ─────────────────────────────────────────────────────────────

def build_genre_vocab(genre_json):
    counts = Counter(v['coarse'] for v in genre_json.values() if v.get('coarse'))
    valid  = {g for g, c in counts.items() if c >= MIN_GENRE_COUNT}
    genres = sorted(valid - {'other'}) + ['other']
    genre2id = {g: i for i, g in enumerate(genres)}
    song2genre = {}
    for sid, info in genre_json.items():
        coarse = info.get('coarse')
        if coarse is None: continue
        if coarse not in valid: coarse = 'other'
        song2genre[sid] = genre2id[coarse]
    label_counts = Counter(song2genre.values())
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (len(genres) * label_counts[i]) for i in range(len(genres))],
        dtype=torch.float,
    )
    return song2genre, genre2id, genres, weights


# ── Build global graph ────────────────────────────────────────────────────────

def build_graph(songs, song2genre, id2genre, class_weights):
    """Build split indices and global HeteroData."""
    rng = random.Random(42)
    indices = list(range(len(songs)))
    rng.shuffle(indices)
    n_train = int(len(indices) * 0.70)
    n_val   = int(len(indices) * 0.15)
    train_idx = set(indices[:n_train])
    val_idx   = set(indices[n_train:n_train + n_val])
    test_idx  = set(indices[n_train + n_val:])

    print(f'Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test songs')
    print('Building global heterogeneous graph (key-normalized)...')

    split_indices = {'train': list(train_idx), 'val': list(val_idx), 'test': list(test_idx)}
    data = build_global_heterograph(songs, split_indices, key_normalize=True)

    # Attach genre labels to song nodes
    genre_labels = torch.full((len(songs),), -1, dtype=torch.long)
    for i, song in enumerate(songs):
        sid = song.get('song_id', '')
        if sid in song2genre:
            genre_labels[i] = song2genre[sid]
    data['song'].genre_label = genre_labels

    n_occ   = data['occ'].x.shape[0]
    n_chord = data['chord'].x.shape[0]
    n_sec   = data['sec'].x.shape[0]
    print(f'  occ nodes:   {n_occ:,}')
    print(f'  chord nodes: {n_chord} (global shared)')
    print(f'  sec nodes:   {n_sec:,}')
    print(f'  song nodes:  {len(songs)}')
    print(f'  edge types:  {[et[1] for et in data.edge_types]}')

    return data, split_indices


# ── Training ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, mask_key='val'):
    model.eval()
    mask      = data['occ'][f'{mask_key}_mask'].to(DEVICE)
    labels    = data['occ'].y.to(DEVICE)
    x_dict    = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    ei_dict   = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}

    chord_logits, genre_logits = model.forward_with_genre(x_dict, ei_dict)

    flat_logits = chord_logits[mask]
    flat_labels = labels[mask]
    valid       = flat_labels != -100
    flat_logits = flat_logits[valid]
    flat_labels = flat_labels[valid]

    chord_ce   = F.cross_entropy(flat_logits, flat_labels).item()
    top1 = (flat_logits.argmax(1) == flat_labels).float().mean().item()
    top5 = (flat_logits.topk(5, dim=1).indices == flat_labels.unsqueeze(1)).any(1).float().mean().item()

    # Genre accuracy
    song_mask   = data['song'][f'{mask_key}_mask'].to(DEVICE)
    genre_label = data['song'].genre_label.to(DEVICE)
    known       = song_mask & (genre_label != -1)
    genre_acc   = 0.0
    if known.any():
        genre_acc = (genre_logits[known].argmax(1) == genre_label[known]).float().mean().item()

    return {'top1': top1, 'top5': top5, 'ce': chord_ce, 'genre_acc': genre_acc}


def train(model, data, class_weights):
    model = model.to(DEVICE)
    genre_wt = class_weights.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    x_dict  = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    ei_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    labels  = data['occ'].y.to(DEVICE)
    train_occ_mask  = data['occ'].train_mask.to(DEVICE)
    train_song_mask = data['song'].train_mask.to(DEVICE)
    genre_label     = data['song'].genre_label.to(DEVICE)

    best_val_ce = float('inf')
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        chord_logits, genre_logits = model.forward_with_genre(x_dict, ei_dict)

        # Chord loss on train occs
        flat_l = chord_logits[train_occ_mask]
        flat_y = labels[train_occ_mask]
        valid  = flat_y != -100
        chord_loss = F.cross_entropy(flat_l[valid], flat_y[valid])

        # Genre loss on train songs with known labels
        known = train_song_mask & (genre_label != -1)
        if known.any():
            genre_loss = F.cross_entropy(genre_logits[known], genre_label[known], weight=genre_wt)
        else:
            genre_loss = torch.tensor(0.0, device=DEVICE)

        loss = chord_loss + GENRE_WEIGHT * genre_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            val = evaluate(model, data, 'val')
            row = {'epoch': epoch, 'chord_loss': chord_loss.item(),
                   **{f'val_{k}': v for k, v in val.items()}}
            history.append(row)
            if val['ce'] < best_val_ce:
                best_val_ce = val['ce']
                torch.save(model.state_dict(), f'{RESULTS_DIR}/global_gnn_best.pt')
            print(f'  Epoch {epoch:3d} | chord_ce={val["ce"]:.4f} '
                  f'| top1={val["top1"]:.3f} | genre_acc={val["genre_acc"]:.3f}')

    return history


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curves(history, save_path):
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [r['val_ce'] for r in history]); axes[0].set_title('Val chord CE')
    axes[1].plot(epochs, [r['val_top1'] for r in history]); axes[1].set_title('Val top-1 (chord)')
    axes[2].plot(epochs, [r['val_genre_acc'] for r in history]); axes[2].set_title('Val genre acc')
    for ax in axes: ax.set_xlabel('Epoch')
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f'Saved {save_path}')


@torch.no_grad()
def plot_umap(model, data, id2genre, save_path):
    from umap import UMAP
    model.eval()
    x_dict  = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    ei_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    h = model._encode(x_dict, ei_dict)
    song_embs = h['song'].cpu().numpy()
    genre_labels = data['song'].genre_label.numpy()

    coords = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(song_embs)
    known  = genre_labels != -1
    cmap   = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(9, 7))
    for gid, gname in enumerate(id2genre):
        sel = (genre_labels == gid) & known
        if sel.any():
            ax.scatter(coords[sel, 0], coords[sel, 1],
                       label=gname, alpha=0.6, s=18, color=cmap(gid))
    ax.legend(markerscale=2, fontsize=9)
    ax.set_title('UMAP — global key-normalized GNN (song nodes, colored by genre)')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f'Saved {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    with open(GENRE_PATH) as f:
        genre_json = json.load(f)
    song2genre, genre2id, id2genre, class_weights = build_genre_vocab(genre_json)
    print(f'Genre classes ({len(id2genre)}): {id2genre}')

    print('Loading songs...')
    songs = load_all_songs(DATA_DIR)
    print(f'  {len(songs)} songs')

    data, split_indices = build_graph(songs, song2genre, id2genre, class_weights)

    model = GlobalMusicGNN(hidden_dim=128, num_layers=3, dropout=0.3, num_genres=len(id2genre))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'GlobalMusicGNN parameters: {n_params:,}')

    print(f'\n=== Training global key-normalized GNN ===')
    history = train(model, data, class_weights)

    model.load_state_dict(torch.load(f'{RESULTS_DIR}/global_gnn_best.pt', map_location=DEVICE))
    test = evaluate(model, data, 'test')
    print(f'\n=== Test Results ===')
    print(f'  Chord top-1:  {test["top1"]:.4f}')
    print(f'  Chord top-5:  {test["top5"]:.4f}')
    print(f'  Chord CE:     {test["ce"]:.4f}')
    print(f'  Genre top-1:  {test["genre_acc"]:.4f}')

    with open(f'{RESULTS_DIR}/global_gnn_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(f'{RESULTS_DIR}/global_gnn_genre_acc.json', 'w') as f:
        json.dump({'test': test}, f, indent=2)

    plot_curves(history, f'{RESULTS_DIR}/global_gnn_training_curves.png')

    print('\nGenerating UMAP...')
    plot_umap(model, data, id2genre, f'{RESULTS_DIR}/global_gnn_genre_umap.png')

    print(f'\nDone. All outputs in {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
