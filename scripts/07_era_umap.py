"""
Script 07 — Era-proxy UMAP of song embeddings

Loads the causal GNN checkpoint, extracts mean-pooled occ embeddings per song,
joins with chart_date from the Billboard index CSV, bins by decade, and plots
a UMAP colored by era.

Run from project root:
    python scripts/07_era_umap.py

Outputs:
    results/era_umap.png
    results/song_embeddings.json   (song_id, era, embedding for downstream use)
"""

import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap

from src.dataset   import build_and_cache
from src.model     import MusicHeteroGNN
from src.train     import collect_occ_embeddings

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
INDEX_CSV     = 'data/billboard-2.0-index.csv'
RESULTS_DIR   = 'results'
CHECKPOINT    = 'results/causal_gnn_best.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


ERA_COLORS = {
    '1950s': '#e41a1c',
    '1960s': '#ff7f00',
    '1970s': '#4daf4a',
    '1980s': '#377eb8',
    '1990s': '#984ea3',
    '2000s': '#a65628',
    'unknown': '#999999',
}


def chart_date_to_era(date_str):
    try:
        year = int(str(date_str)[:4])
        decade = (year // 10) * 10
        return f'{decade}s'
    except (ValueError, TypeError):
        return 'unknown'


def main():
    # ── Load graphs ────────────────────────────────────────────────────────
    print('Loading graphs...')
    graphs = build_and_cache(DATA_DIR, PROCESSED_DIR)
    print(f'  {len(graphs)} graphs loaded')

    # ── Load index CSV → song_id → era mapping ────────────────────────────
    index_df = pd.read_csv(INDEX_CSV)
    # song_id is zero-padded 4-digit string matching folder names
    index_df['song_id'] = index_df['id'].apply(lambda x: f'{int(x):04d}')
    index_df['era']     = index_df['chart_date'].apply(chart_date_to_era)
    id_to_era = dict(zip(index_df['song_id'], index_df['era']))

    # ── Load causal GNN ────────────────────────────────────────────────────
    print('Loading model...')
    model = MusicHeteroGNN(
        hidden_dim=128,
        num_layers=3,
        dropout=0.0,          # no dropout at eval
        use_prev_edges=False,
        use_chord_in_occ=True,
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # ── Extract embeddings ─────────────────────────────────────────────────
    print('Extracting song embeddings...')
    embeddings, song_ids, _ = collect_occ_embeddings(model, graphs, device=DEVICE)
    # embeddings: [N_songs, 128]
    emb_np = embeddings.numpy()
    print(f'  Got {emb_np.shape[0]} song embeddings of dim {emb_np.shape[1]}')

    # ── Attach era labels ──────────────────────────────────────────────────
    eras = [id_to_era.get(sid, 'unknown') for sid in song_ids]
    era_counts = pd.Series(eras).value_counts()
    print('Era distribution:')
    for era, count in sorted(era_counts.items()):
        print(f'  {era}: {count}')

    # ── UMAP ───────────────────────────────────────────────────────────────
    print('Running UMAP...')
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_xy = reducer.fit_transform(emb_np)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    for era in sorted(ERA_COLORS):
        mask = np.array([e == era for e in eras])
        if mask.sum() == 0:
            continue
        color = ERA_COLORS[era]
        ax.scatter(
            umap_xy[mask, 0], umap_xy[mask, 1],
            c=color, label=f'{era} (n={mask.sum()})',
            s=25, alpha=0.75, linewidths=0,
        )

    ax.set_title('Song Embedding UMAP — Colored by Era\n(Causal GNN, mean-pooled occ embeddings)', fontsize=13)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    out_path = f'{RESULTS_DIR}/era_umap.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved {out_path}')

    # ── Save embeddings + metadata for downstream genre analysis ──────────
    records = [
        {'song_id': sid, 'era': era, 'umap_x': float(xy[0]), 'umap_y': float(xy[1])}
        for sid, era, xy in zip(song_ids, eras, umap_xy)
    ]
    with open(f'{RESULTS_DIR}/song_embeddings_meta.json', 'w') as f:
        json.dump(records, f, indent=2)

    # Also save raw embeddings as numpy for reuse in genre script
    np.save(f'{RESULTS_DIR}/song_embeddings.npy', emb_np)
    with open(f'{RESULTS_DIR}/song_ids.json', 'w') as f:
        json.dump(list(song_ids), f)

    print(f'Saved song_embeddings.npy and song_ids.json for downstream genre analysis')
    print('Done.')


if __name__ == '__main__':
    main()
