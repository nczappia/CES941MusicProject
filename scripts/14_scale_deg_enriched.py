"""
Script 14 — Scale-degree enriched multi-task GNN

Same config as script 10 (causal v2, chord-in-occ, 3 layers, multi-task genre)
but adds scale_deg nodes (12 chromatic degrees relative to annotated tonic) so
the GNN can learn I→IV→V patterns in key-agnostic form.

Graph schema additions (already in src/graph.py):
  scale_deg nodes  : 12 nodes per graph (eye(12) features)
  chord_degree edges: chord → scale_deg (root's chromatic degree from tonic)
  degree_rev edges  : scale_deg → chord (reverse)

Key question: do scale degree nodes improve chord prediction and/or genre acc
vs the baseline enriched run (script 10)?

Outputs:
    results/scale_deg_best.pt
    results/scale_deg_history.json
    results/scale_deg_training_curves.png
    results/scale_deg_genre_umap.png
    results/scale_deg_genre_acc.json

Run from project root:
    source venv/bin/activate && python scripts/14_scale_deg_enriched.py
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from torch_geometric.data import Batch

from src.dataset import get_splits
from src.model   import MusicHeteroGNN
from src.train   import evaluate_gnn, make_batch

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'
GENRE_PATH    = 'data/genre_labels.json'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

GENRE_WEIGHT    = 0.5
EPOCHS          = 60
LR              = 1e-3
BATCH_SIZE      = 16
MIN_GENRE_COUNT = 10


# ── Genre helpers ─────────────────────────────────────────────────────────────

def build_genre_vocab(genre_json):
    counts   = Counter(v['coarse'] for v in genre_json.values() if v.get('coarse'))
    valid    = {g for g, c in counts.items() if c >= MIN_GENRE_COUNT}
    genres   = sorted(valid - {'other'}) + ['other']
    genre2id = {g: i for i, g in enumerate(genres)}

    song2genre = {}
    for sid, info in genre_json.items():
        coarse = info.get('coarse')
        if coarse is None:
            continue
        if coarse not in valid:
            coarse = 'other'
        song2genre[sid] = genre2id[coarse]

    label_counts = Counter(song2genre.values())
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (len(genres) * label_counts[i]) for i in range(len(genres))],
        dtype=torch.float,
    )
    return song2genre, genre2id, genres, weights


def attach_genre_labels(graphs, song2genre):
    for g in graphs:
        sid = getattr(g, 'song_id', '')
        g.genre_label = song2genre.get(sid, -1)


# ── Training ─────────────────────────────────────────────────────────────────

def train_multitask(model, train_graphs, val_graphs, class_weights):
    model = model.to(DEVICE)
    genre_weight_tensor = class_weights.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(train_graphs)

        epoch_chord_loss = 0.0
        epoch_genre_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_graphs), BATCH_SIZE):
            batch_graphs = train_graphs[start:start + BATCH_SIZE]
            batch = make_batch(batch_graphs, DEVICE)
            occ_batch = batch['occ'].batch

            optimizer.zero_grad()
            chord_logits, genre_logits = model.forward_with_genre(
                batch.x_dict, batch.edge_index_dict, occ_batch
            )

            labels = batch['occ'].y
            mask   = labels != -100
            chord_loss = F.cross_entropy(chord_logits[mask], labels[mask])

            genre_labels = torch.tensor(
                [g.genre_label for g in batch_graphs],
                dtype=torch.long, device=DEVICE
            )
            genre_mask = genre_labels != -1
            if genre_mask.any():
                genre_loss = F.cross_entropy(
                    genre_logits[genre_mask], genre_labels[genre_mask],
                    weight=genre_weight_tensor,
                )
            else:
                genre_loss = torch.tensor(0.0, device=DEVICE)

            loss = chord_loss + GENRE_WEIGHT * genre_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_chord_loss += chord_loss.item()
            epoch_genre_loss += genre_loss.item()
            n_batches += 1

        scheduler.step()

        val_metrics = evaluate_gnn(model, val_graphs, device=DEVICE)
        genre_val   = eval_genre(model, val_graphs)

        row = {
            'epoch':             epoch,
            'chord_loss':        epoch_chord_loss / max(n_batches, 1),
            'genre_loss':        epoch_genre_loss / max(n_batches, 1),
            'val_top1_acc':      val_metrics['top1_acc'],
            'val_top5_acc':      val_metrics['top5_acc'],
            'val_cross_entropy': val_metrics['cross_entropy'],
            'val_genre_acc':     genre_val,
        }
        history.append(row)

        if val_metrics['cross_entropy'] < best_val_loss:
            best_val_loss = val_metrics['cross_entropy']
            torch.save(model.state_dict(), f'{RESULTS_DIR}/scale_deg_best.pt')

        if epoch % 5 == 0:
            print(f'  Epoch {epoch:3d} | chord_ce={val_metrics["cross_entropy"]:.4f} '
                  f'| top1={val_metrics["top1_acc"]:.3f} '
                  f'| genre_acc={genre_val:.3f}')

    return history


@torch.no_grad()
def eval_genre(model, graphs, batch_size=32):
    model.eval()
    hits = 0
    total = 0
    for start in range(0, len(graphs), batch_size):
        batch_graphs = graphs[start:start + batch_size]
        batch = make_batch(batch_graphs, DEVICE)
        occ_batch = batch['occ'].batch
        _, genre_logits = model.forward_with_genre(
            batch.x_dict, batch.edge_index_dict, occ_batch
        )
        genre_labels = torch.tensor(
            [g.genre_label for g in batch_graphs],
            dtype=torch.long, device=DEVICE
        )
        mask = genre_labels != -1
        if mask.any():
            preds = genre_logits[mask].argmax(dim=1)
            hits  += (preds == genre_labels[mask]).sum().item()
            total += mask.sum().item()
    return hits / total if total > 0 else 0.0


@torch.no_grad()
def eval_genre_per_class(model, graphs, id2genre):
    hits   = Counter()
    totals = Counter()
    model.eval()
    for g in graphs:
        if g.genre_label == -1:
            continue
        g2 = g.to(DEVICE)
        occ_batch = torch.zeros(g2['occ'].x.shape[0], dtype=torch.long, device=DEVICE)
        _, genre_logits = model.forward_with_genre(
            g2.x_dict, g2.edge_index_dict, occ_batch
        )
        pred = genre_logits.argmax(dim=1).item()
        true = g.genre_label
        totals[true] += 1
        if pred == true:
            hits[true] += 1
    results = {}
    for gid, gname in enumerate(id2genre):
        if totals[gid] > 0:
            results[gname] = {'acc': hits[gid] / totals[gid], 'n': totals[gid]}
    return results


@torch.no_grad()
def collect_song_embeddings(model, graphs):
    from src.train import _get_occ_embeddings
    model.eval()
    embs, labels, sids = [], [], []
    for g in graphs:
        g2 = g.to(DEVICE)
        occ_emb = _get_occ_embeddings(model, g2)
        song_emb = occ_emb.mean(dim=0).cpu()
        embs.append(song_emb)
        labels.append(g.genre_label)
        sids.append(getattr(g, 'song_id', ''))
    return torch.stack(embs).numpy(), labels, sids


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_curves(history, save_path):
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [r['chord_loss'] for r in history], label='chord CE (train)')
    axes[0].plot(epochs, [r['val_cross_entropy'] for r in history], label='chord CE (val)')
    axes[0].set_title('Chord cross-entropy'); axes[0].set_xlabel('Epoch'); axes[0].legend()

    axes[1].plot(epochs, [r['val_top1_acc'] for r in history])
    axes[1].set_title('Val top-1 (chord)'); axes[1].set_xlabel('Epoch')

    axes[2].plot(epochs, [r['val_genre_acc'] for r in history])
    axes[2].set_title('Val genre accuracy'); axes[2].set_xlabel('Epoch')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_genre_umap(embeddings, genre_labels, id2genre, save_path):
    from umap import UMAP
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords  = reducer.fit_transform(embeddings)

    known_mask = np.array(genre_labels) != -1
    coords_k   = coords[known_mask]
    labels_k   = np.array(genre_labels)[known_mask]

    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(9, 7))
    for gid, gname in enumerate(id2genre):
        sel = labels_k == gid
        if sel.any():
            ax.scatter(coords_k[sel, 0], coords_k[sel, 1],
                       label=gname, alpha=0.6, s=18, color=cmap(gid))
    ax.legend(markerscale=2, fontsize=9)
    ax.set_title('UMAP — scale-deg enriched GNN (colored by genre)')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    with open(GENRE_PATH) as f:
        genre_json = json.load(f)
    song2genre, genre2id, id2genre, class_weights = build_genre_vocab(genre_json)
    num_genres = len(id2genre)
    print(f'Genre classes ({num_genres}): {id2genre}')

    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED_DIR)
    attach_genre_labels(train_g, song2genre)
    attach_genre_labels(val_g,   song2genre)
    attach_genre_labels(test_g,  song2genre)

    labeled_train = sum(1 for g in train_g if g.genre_label != -1)
    print(f'Train songs with genre label: {labeled_train}/{len(train_g)}')

    # Verify scale_deg nodes in first graph
    g0 = train_g[0]
    print(f'Scale_deg node count in first graph: {g0["scale_deg"].x.shape[0]}')
    print(f'Edge types: {[et[1] for et in g0.edge_types]}')

    model = MusicHeteroGNN(
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        use_prev_edges=False,
        use_chord_in_occ=True,
        use_scale_deg_edges=True,
        num_genres=num_genres,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Scale-deg enriched model parameters: {n_params:,}')

    print(f'\n=== Training scale-degree enriched GNN (genre_weight={GENRE_WEIGHT}) ===')
    history = train_multitask(model, train_g, val_g, class_weights)

    model.load_state_dict(torch.load(f'{RESULTS_DIR}/scale_deg_best.pt', map_location=DEVICE))

    test_chord      = evaluate_gnn(model, test_g, device=DEVICE)
    test_genre      = eval_genre(model, test_g)
    genre_breakdown = eval_genre_per_class(model, test_g, id2genre)

    print(f'\n=== Test Results ===')
    print(f'  Chord top-1: {test_chord["top1_acc"]:.4f}')
    print(f'  Chord top-5: {test_chord["top5_acc"]:.4f}')
    print(f'  Chord CE:    {test_chord["cross_entropy"]:.4f}')
    print(f'  Genre top-1: {test_genre:.4f}')
    print(f'\n  Per-genre accuracy:')
    for gname, d in sorted(genre_breakdown.items(), key=lambda x: -x[1]['n']):
        print(f'    {gname:<15s}  acc={d["acc"]:.3f}  n={d["n"]}')

    with open(f'{RESULTS_DIR}/scale_deg_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(f'{RESULTS_DIR}/scale_deg_genre_acc.json', 'w') as f:
        json.dump({
            'test_chord': test_chord,
            'test_genre_overall': test_genre,
            'test_genre_per_class': genre_breakdown,
        }, f, indent=2)

    plot_training_curves(history, f'{RESULTS_DIR}/scale_deg_training_curves.png')

    print('\nCollecting embeddings for UMAP...')
    all_graphs = train_g + val_g + test_g
    embs, genre_labels, _ = collect_song_embeddings(model, all_graphs)
    plot_genre_umap(embs, genre_labels, id2genre, f'{RESULTS_DIR}/scale_deg_genre_umap.png')

    print(f'\nDone. All outputs in {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
