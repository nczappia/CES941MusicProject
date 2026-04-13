"""
Script 05 — Generate all final visualizations

Run from project root (after scripts 03 and 04):
    python scripts/05_visualize.py

Outputs:
    results/umap_songs.png           — per-song embedding map coloured by section type
    results/model_comparison.png     — all models side by side
    results/chord_entropy_by_sec.png — prediction entropy per section type
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.dataset   import get_splits
from src.model     import MusicHeteroGNN, NUM_CLASSES
from src.train     import collect_occ_embeddings, evaluate_gnn
from src.visualize import (
    plot_embedding_umap, plot_model_comparison, plot_ablation_bar,
)
from src.vocab     import SECTION_TYPES

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    _, _, test_g = get_splits(DATA_DIR, PROCESSED_DIR)

    # ── Load trained model ────────────────────────────────────────────────
    ckpt = f'{RESULTS_DIR}/gnn_best.pt'
    if not Path(ckpt).exists():
        print(f'Checkpoint not found at {ckpt}. Run 03_train_gnn.py first.')
        return

    model = MusicHeteroGNN(hidden_dim=128, num_layers=2, dropout=0.0)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    # ── UMAP of song embeddings ───────────────────────────────────────────
    print('Computing song embeddings...')
    embs, song_ids, sec_labels = collect_occ_embeddings(model, test_g, device=DEVICE)

    plot_embedding_umap(
        embs, sec_labels, SECTION_TYPES,
        title='Song embeddings (UMAP) — coloured by dominant section type',
        save_path=f'{RESULTS_DIR}/umap_songs.png',
    )

    # ── Full model comparison (needs baseline_results.json) ───────────────
    baseline_file = f'{RESULTS_DIR}/baseline_results.json'
    gnn_file      = f'{RESULTS_DIR}/gnn_test_results.json'

    if Path(baseline_file).exists() and Path(gnn_file).exists():
        with open(baseline_file) as f:
            baseline_res = json.load(f)
        with open(gnn_file) as f:
            gnn_res = json.load(f)

        all_models = {**baseline_res, 'HeteroGNN': gnn_res['test']}
        plot_model_comparison(
            all_models,
            metrics=['top1_acc', 'top5_acc', 'top10_acc'],
            save_path=f'{RESULTS_DIR}/model_comparison.png',
        )
    else:
        print('Baseline or GNN results missing; skipping model comparison plot.')

    # ── Prediction entropy by section type ───────────────────────────────
    print('Computing prediction entropy by section...')
    _plot_entropy_by_section(model, test_g, DEVICE)


@torch.no_grad()
def _plot_entropy_by_section(model, graphs, device):
    import torch.nn.functional as F

    model = model.to(device)
    sec_entropies = defaultdict(list)

    for g in graphs:
        g = g.to(device)
        logits = model(g.x_dict, g.edge_index_dict)       # [N_occ, C]
        probs  = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)   # [N_occ]

        labels = g['occ'].y
        mask   = labels != -100

        sec_feats    = g['sec'].x
        sec_type_ids = sec_feats[:, 2:].argmax(dim=1)
        ei_secrev    = g['sec', 'sec_rev', 'occ'].edge_index
        sec_for_occ  = torch.zeros(labels.shape[0], dtype=torch.long, device=device)
        sec_for_occ[ei_secrev[1]] = ei_secrev[0]
        occ_stypes   = sec_type_ids[sec_for_occ]

        for i in range(labels.shape[0]):
            if not mask[i]:
                continue
            stype = SECTION_TYPES[occ_stypes[i].item()]
            sec_entropies[stype].append(entropy[i].item())

    # Filter to section types with enough samples
    items = [(s, np.mean(v), np.std(v), len(v))
             for s, v in sec_entropies.items() if len(v) >= 20]
    items.sort(key=lambda x: -x[1])

    names   = [f'{s}' for s, *_ in items]
    means   = [m for _, m, *_ in items]
    stds    = [sd for _, _, sd, _ in items]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.5)))
    ax.barh(names, means, xerr=stds, color='steelblue',
            edgecolor='white', height=0.6, capsize=3)
    ax.set_xlabel('Mean prediction entropy (nats)')
    ax.set_title('Chord prediction uncertainty by section type')
    plt.tight_layout()
    out = f'{RESULTS_DIR}/chord_entropy_by_sec.png'
    plt.savefig(out, dpi=150)
    print(f'Saved {out}')
    plt.close()


if __name__ == '__main__':
    main()
