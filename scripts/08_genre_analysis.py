"""
Script 08 — Genre-colored UMAP and chord usage analysis

Requires:
    data/genre_labels.json       (from scripts/fetch_genres.py)
    results/song_embeddings.npy  (from scripts/07_era_umap.py)
    results/song_ids.json        (from scripts/07_era_umap.py)
    data/processed/graphs.pkl    (cached graphs)

Run from project root:
    python scripts/08_genre_analysis.py

Outputs:
    results/genre_umap.png              — UMAP colored by coarse genre
    results/genre_chord_heatmap.png     — top chord types per genre (heatmap)
    results/genre_progression_bars.png  — most common 2-chord transitions per genre
    results/genre_analysis.json         — summary stats
"""

import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap
from collections import Counter, defaultdict
from pathlib import Path

from src.vocab import VOCAB_SIZE, N_CHORD_ID

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'
GENRE_FILE    = 'data/genre_labels.json'
EMB_FILE      = 'results/song_embeddings.npy'
IDS_FILE      = 'results/song_ids.json'

# Coarse genre display order and colors
GENRE_ORDER = ['rock', 'pop', 'soul_r&b', 'country', 'jazz', 'blues',
               'hip_hop', 'disco_dance', 'folk', 'other']
GENRE_COLORS = {
    'rock':        '#e41a1c',
    'pop':         '#ff7f00',
    'soul_r&b':    '#984ea3',
    'country':     '#4daf4a',
    'jazz':        '#377eb8',
    'blues':       '#a65628',
    'hip_hop':     '#f781bf',
    'disco_dance': '#ffff33',
    'folk':        '#999999',
    'other':       '#dddddd',
    'unknown':     '#cccccc',
}

# Human-readable chord label from vocab id
def chord_id_to_str(cid: int) -> str:
    if cid == N_CHORD_ID:
        return 'N'
    roots  = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
    quals  = ['maj','min','dim','aug','sus','oth']
    cmplx  = ['','7']
    root = cid // 12
    qual = (cid % 12) // 2
    cx   = cid % 2
    return f'{roots[root]}:{quals[qual]}{cmplx[cx]}'


def load_graphs_with_genre(genre_labels: dict, graphs, song_ids: list):
    """Zip graphs with their genre label, skipping unknowns."""
    labeled = []
    for g, sid in zip(graphs, song_ids):
        info = genre_labels.get(sid)
        if info and info['coarse'] not in ('unknown', 'other'):
            labeled.append((g, info['coarse']))
    return labeled


# ── Plot 1: Genre-colored UMAP ────────────────────────────────────────────

def plot_genre_umap(emb_np, song_ids, genre_labels, umap_xy=None):
    genres = [genre_labels.get(sid, {}).get('coarse', 'unknown') for sid in song_ids]

    if umap_xy is None:
        print('Running UMAP...')
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_xy = reducer.fit_transform(emb_np)

    fig, ax = plt.subplots(figsize=(11, 8))
    for genre in GENRE_ORDER + ['unknown']:
        mask = np.array([g == genre for g in genres])
        if mask.sum() == 0:
            continue
        color = GENRE_COLORS.get(genre, '#cccccc')
        ax.scatter(
            umap_xy[mask, 0], umap_xy[mask, 1],
            c=color, label=f'{genre} (n={mask.sum()})',
            s=28, alpha=0.8, linewidths=0,
        )

    ax.set_title('Song Embedding UMAP — Colored by Genre\n(Causal GNN, mean-pooled occ embeddings)', fontsize=13)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85, ncol=2)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out = f'{RESULTS_DIR}/genre_umap.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')
    return umap_xy


# ── Plot 2: Top chord usage per genre (heatmap) ───────────────────────────

def plot_chord_heatmap(labeled_graphs, top_n=15):
    """
    For each genre, compute the normalized frequency of each chord type.
    Plot as a heatmap: genres × top chords.
    """
    genre_chord_counts = defaultdict(Counter)
    genre_totals       = Counter()

    for g, genre in labeled_graphs:
        chord_ids = g['occ'].y.tolist()
        for cid in chord_ids:
            if cid == -100:
                continue
            genre_chord_counts[genre][cid] += 1
            genre_totals[genre] += 1

    # Find the globally most common chord IDs across all genres
    global_counts = Counter()
    for counts in genre_chord_counts.values():
        global_counts.update(counts)
    top_chords = [cid for cid, _ in global_counts.most_common(top_n)]
    top_labels = [chord_id_to_str(c) for c in top_chords]

    genres_present = [g for g in GENRE_ORDER if g in genre_chord_counts]

    # Build matrix: [genres, top_chords] — row-normalized frequencies
    mat = np.zeros((len(genres_present), top_n))
    for i, genre in enumerate(genres_present):
        total = genre_totals[genre]
        for j, cid in enumerate(top_chords):
            mat[i, j] = genre_chord_counts[genre][cid] / max(total, 1)

    fig, ax = plt.subplots(figsize=(14, max(4, len(genres_present) * 0.7 + 1)))
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(genres_present)))
    ax.set_yticklabels(genres_present, fontsize=10)
    ax.set_title(f'Chord Usage Frequency by Genre (top {top_n} chords, row-normalized)', fontsize=12)

    plt.colorbar(im, ax=ax, label='Fraction of chord occurrences')
    plt.tight_layout()
    out = f'{RESULTS_DIR}/genre_chord_heatmap.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')


# ── Plot 3: Top bigram transitions per genre ──────────────────────────────

def plot_transition_bars(labeled_graphs, top_genres=5, top_k=8):
    """
    For the top N genres by song count, show the most common 2-chord transitions.
    """
    genre_bigrams = defaultdict(Counter)
    genre_song_counts = Counter()

    for g, genre in labeled_graphs:
        chord_ids = [c for c in g['occ'].y.tolist() if c != -100]
        genre_song_counts[genre] += 1
        for a, b in zip(chord_ids[:-1], chord_ids[1:]):
            if a != N_CHORD_ID and b != N_CHORD_ID:
                label = f'{chord_id_to_str(a)} → {chord_id_to_str(b)}'
                genre_bigrams[genre][label] += 1

    genres_to_show = [g for g, _ in genre_song_counts.most_common(top_genres)
                      if g in GENRE_ORDER][:top_genres]

    fig, axes = plt.subplots(1, len(genres_to_show),
                             figsize=(4.5 * len(genres_to_show), 5), sharey=False)
    if len(genres_to_show) == 1:
        axes = [axes]

    for ax, genre in zip(axes, genres_to_show):
        top = genre_bigrams[genre].most_common(top_k)
        labels = [t[0] for t in top]
        counts = [t[1] for t in top]
        total = sum(genre_bigrams[genre].values())
        freqs = [c / total for c in counts]

        color = GENRE_COLORS.get(genre, '#888888')
        bars = ax.barh(range(len(labels)), freqs, color=color, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'{genre}\n(n={genre_song_counts[genre]} songs)', fontsize=10)
        ax.set_xlabel('Relative frequency', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)

    plt.suptitle('Most Common Chord Transitions by Genre', fontsize=13, y=1.02)
    plt.tight_layout()
    out = f'{RESULTS_DIR}/genre_progression_bars.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # ── Load prerequisites ────────────────────────────────────────────────
    if not Path(GENRE_FILE).exists():
        print(f'ERROR: {GENRE_FILE} not found. Run scripts/fetch_genres.py first.')
        return
    if not Path(EMB_FILE).exists():
        print(f'ERROR: {EMB_FILE} not found. Run scripts/07_era_umap.py first.')
        return

    with open(GENRE_FILE) as f:
        genre_labels = json.load(f)
    emb_np   = np.load(EMB_FILE)
    with open(IDS_FILE) as f:
        song_ids = json.load(f)

    print(f'Loaded {len(genre_labels)} genre labels, {len(song_ids)} song embeddings')

    # Genre distribution
    coarse_counts = Counter(v['coarse'] for v in genre_labels.values())
    print('Genre distribution:')
    for g, c in sorted(coarse_counts.items(), key=lambda x: -x[1]):
        print(f'  {g:<15s}: {c}')

    # ── Load graphs for chord analysis ────────────────────────────────────
    cache_path = Path(PROCESSED_DIR) / 'graphs.pkl'
    with open(cache_path, 'rb') as f:
        all_graphs = pickle.load(f)

    # Build song_id → graph mapping
    sid_to_graph = {getattr(g, 'song_id', ''): g for g in all_graphs}
    graphs_in_order = [sid_to_graph.get(sid) for sid in song_ids]

    labeled = load_graphs_with_genre(genre_labels,
                                     [g for g in graphs_in_order if g is not None],
                                     [sid for sid, g in zip(song_ids, graphs_in_order) if g is not None])
    print(f'\n{len(labeled)} songs with known genre labels for chord analysis')

    # ── UMAP ─────────────────────────────────────────────────────────────
    print('\nPlotting genre UMAP...')
    # Re-use UMAP coordinates from era script if available, else recompute
    meta_path = Path(f'{RESULTS_DIR}/song_embeddings_meta.json')
    umap_xy = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta_by_id = {m['song_id']: m for m in meta}
        umap_xy_list = [meta_by_id[sid] for sid in song_ids if sid in meta_by_id]
        if len(umap_xy_list) == len(song_ids):
            umap_xy = np.array([[m['umap_x'], m['umap_y']] for m in umap_xy_list])
            print('  Reusing UMAP coordinates from era script')

    umap_xy = plot_genre_umap(emb_np, song_ids, genre_labels, umap_xy=umap_xy)

    # ── Chord heatmap ─────────────────────────────────────────────────────
    print('Plotting chord heatmap...')
    plot_chord_heatmap(labeled)

    # ── Transition bars ───────────────────────────────────────────────────
    print('Plotting transition bars...')
    plot_transition_bars(labeled)

    # ── Save summary stats ────────────────────────────────────────────────
    summary = {
        'genre_counts': dict(coarse_counts),
        'labeled_songs': len(labeled),
        'total_songs': len(song_ids),
        'match_rate': len([v for v in genre_labels.values()
                           if v['source'] not in ('not_found', 'no_metadata')]) / max(len(genre_labels), 1),
    }
    with open(f'{RESULTS_DIR}/genre_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nAll genre analysis outputs saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
