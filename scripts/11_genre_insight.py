"""
Script 11 — Genre embedding insight analysis

Using the trained multi-task model, produces:
  results/genre_confusion_matrix.png  — predicted vs true genre
  results/genre_centroid_similarity.png — cosine sim between genre mean embeddings
  results/genre_typical_songs.json    — top-3 most genre-typical songs per genre

Run from project root:
    source venv/bin/activate && python scripts/11_genre_insight.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from collections import Counter, defaultdict

from src.dataset import get_splits
from src.model   import MusicHeteroGNN
from src.train   import _get_occ_embeddings, make_batch

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'
GENRE_PATH    = 'data/genre_labels.json'
CKPT_PATH     = 'results/multitask_best.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_GENRE_COUNT = 10


# ── Reuse vocab builder from script 10 ───────────────────────────────────────

def build_genre_vocab(genre_json):
    counts = Counter(v['coarse'] for v in genre_json.values() if v.get('coarse'))
    valid  = {g for g, c in counts.items() if c >= MIN_GENRE_COUNT}
    genres = sorted(valid - {'other'}) + ['other']
    genre2id = {g: i for i, g in enumerate(genres)}
    song2genre = {}
    for sid, info in genre_json.items():
        coarse = info.get('coarse')
        if coarse is None:
            continue
        if coarse not in valid:
            coarse = 'other'
        song2genre[sid] = genre2id[coarse]
    return song2genre, genre2id, genres


def attach_genre_labels(graphs, song2genre):
    for g in graphs:
        sid = getattr(g, 'song_id', '')
        g.genre_label = song2genre.get(sid, -1)


# ── Collect per-song embeddings + predictions ─────────────────────────────────

@torch.no_grad()
def collect_all(model, graphs):
    model.eval()
    embs, true_labels, pred_labels, song_ids = [], [], [], []
    for g in graphs:
        if g.genre_label == -1:
            continue
        g2 = g.to(DEVICE)
        occ_emb  = _get_occ_embeddings(model, g2)       # [N_occ, H]
        song_emb = occ_emb.mean(dim=0)                  # [H]

        occ_batch = torch.zeros(g2['occ'].x.shape[0], dtype=torch.long, device=DEVICE)
        _, genre_logits = model.forward_with_genre(
            g2.x_dict, g2.edge_index_dict, occ_batch
        )
        pred = genre_logits.argmax(dim=1).item()

        embs.append(song_emb.cpu())
        true_labels.append(g.genre_label)
        pred_labels.append(pred)
        song_ids.append(getattr(g, 'song_id', ''))

    return torch.stack(embs).numpy(), true_labels, pred_labels, song_ids


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(true_labels, pred_labels, id2genre, save_path):
    n = len(id2genre)
    mat = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        mat[t, p] += 1

    # Row-normalize to show rates
    row_sums = mat.sum(axis=1, keepdims=True).clip(min=1)
    mat_norm = mat / row_sums

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat_norm, vmin=0, vmax=1, cmap='Blues')
    plt.colorbar(im, ax=ax, label='Fraction of true-class predictions')

    ax.set_xticks(range(n)); ax.set_xticklabels(id2genre, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(id2genre)
    ax.set_xlabel('Predicted genre')
    ax.set_ylabel('True genre')
    ax.set_title('Genre confusion matrix (row-normalized)')

    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0:
                color = 'white' if mat_norm[i, j] > 0.5 else 'black'
                ax.text(j, i, str(mat[i, j]), ha='center', va='center',
                        fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved {save_path}')


# ── Genre centroid similarity ─────────────────────────────────────────────────

def plot_centroid_similarity(embeddings, true_labels, id2genre, save_path):
    n = len(id2genre)
    centroids = []
    for gid in range(n):
        mask = np.array(true_labels) == gid
        if mask.any():
            c = embeddings[mask].mean(axis=0)
            centroids.append(c / (np.linalg.norm(c) + 1e-8))
        else:
            centroids.append(np.zeros(embeddings.shape[1]))
    centroids = np.stack(centroids)

    # Cosine similarity matrix
    sim = centroids @ centroids.T

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Cosine similarity')

    ax.set_xticks(range(n)); ax.set_xticklabels(id2genre, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(id2genre)
    ax.set_title('Genre centroid cosine similarity\n(embedding space — multi-task model)')

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{sim[i,j]:.2f}', ha='center', va='center', fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved {save_path}')

    # Print most/least similar pairs
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((sim[i, j], id2genre[i], id2genre[j]))
    pairs.sort(reverse=True)
    print('\nMost similar genre pairs (embedding space):')
    for s, a, b in pairs[:5]:
        print(f'  {a} ↔ {b}: {s:.3f}')
    print('Least similar genre pairs:')
    for s, a, b in pairs[-3:]:
        print(f'  {a} ↔ {b}: {s:.3f}')

    return sim, id2genre


# ── Most genre-typical songs ──────────────────────────────────────────────────

def find_typical_songs(embeddings, true_labels, song_ids, id2genre, genre_json, save_path):
    results = {}
    for gid, gname in enumerate(id2genre):
        mask = np.array(true_labels) == gid
        if not mask.any():
            continue
        genre_embs = embeddings[mask]
        genre_sids = [s for s, m in zip(song_ids, mask) if m]

        centroid = genre_embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8

        # Cosine similarity to centroid
        norms = np.linalg.norm(genre_embs, axis=1, keepdims=True) + 1e-8
        sims  = (genre_embs / norms) @ centroid
        top_idx = sims.argsort()[::-1][:3]

        results[gname] = []
        for i in top_idx:
            sid = genre_sids[i]
            info = genre_json.get(sid, {})
            results[gname].append({
                'song_id':   sid,
                'title':     info.get('title', '?'),
                'artist':    info.get('artist', '?'),
                'similarity': float(sims[i]),
            })

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {save_path}')

    print('\nMost genre-typical songs:')
    for gname, songs in results.items():
        print(f'\n  {gname.upper()}')
        for s in songs:
            print(f'    {s["artist"]} — {s["title"]}  (sim={s["similarity"]:.3f})')

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(GENRE_PATH) as f:
        genre_json = json.load(f)
    song2genre, genre2id, id2genre = build_genre_vocab(genre_json)
    num_genres = len(id2genre)

    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED_DIR)
    all_graphs = train_g + val_g + test_g
    attach_genre_labels(all_graphs, song2genre)

    model = MusicHeteroGNN(
        hidden_dim=128, num_layers=3, dropout=0.3,
        use_prev_edges=False, use_chord_in_occ=True,
        num_genres=num_genres,
    )
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f'Loaded checkpoint: {CKPT_PATH}')

    print('Collecting embeddings and predictions...')
    embs, true_labels, pred_labels, song_ids = collect_all(model, all_graphs)
    print(f'  {len(embs)} songs with genre labels')

    plot_confusion_matrix(true_labels, pred_labels, id2genre,
                          f'{RESULTS_DIR}/genre_confusion_matrix.png')

    plot_centroid_similarity(embs, true_labels, id2genre,
                             f'{RESULTS_DIR}/genre_centroid_similarity.png')

    find_typical_songs(embs, true_labels, song_ids, id2genre, genre_json,
                       f'{RESULTS_DIR}/genre_typical_songs.json')


if __name__ == '__main__':
    main()
