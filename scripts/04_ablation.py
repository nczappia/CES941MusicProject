"""
Script 04 — Ablation study

Trains one model per ablation condition and compares test metrics.

Run from project root:
    python scripts/04_ablation.py

Outputs:
    results/ablation_results.json
    results/ablation_bar.png
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path

from src.dataset   import get_splits
from src.model     import MusicHeteroGNN
from src.train     import train_gnn, evaluate_gnn
from src.visualize import plot_ablation_bar

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Each condition: (display_name, model kwargs override)
ABLATION_CONDITIONS = [
    ('Full model',               {}),
    ('No sequence edges',        {'use_seq_edges':     False}),
    ('No instance_of edges',     {'use_inst_edges':    False}),
    ('No section edges',         {'use_section_edges': False}),
    ('No sec→sec edges',         {'use_sec_seq_edges': False}),
    ('No section features',      {'use_sec_features':  False}),
]

BASE_MODEL_KWARGS = dict(hidden_dim=128, num_layers=2, dropout=0.3)
TRAIN_KWARGS      = dict(epochs=50, lr=1e-3, weight_decay=1e-4,
                         batch_size=16, device=DEVICE)


def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED_DIR)

    ablation_results = {}

    for name, overrides in ABLATION_CONDITIONS:
        print(f'\n=== Ablation: {name} ===')
        kwargs = {**BASE_MODEL_KWARGS, **overrides}
        model  = MusicHeteroGNN(**kwargs)

        ckpt = f'{RESULTS_DIR}/ablation_{name.lower().replace(" ", "_")}.pt'
        history = train_gnn(
            model, train_g, val_g,
            checkpoint_path=ckpt,
            **TRAIN_KWARGS,
        )

        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        metrics = evaluate_gnn(model, test_g, device=DEVICE)

        print(f'  top1={metrics["top1_acc"]:.3f} | top5={metrics["top5_acc"]:.3f} '
              f'| CE={metrics["cross_entropy"]:.4f}')
        ablation_results[name] = metrics

    # Save
    with open(f'{RESULTS_DIR}/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print(f'\nSaved ablation_results.json')

    # Plot
    plot_ablation_bar(ablation_results, metric='top1_acc',
                      save_path=f'{RESULTS_DIR}/ablation_bar_top1.png')
    plot_ablation_bar(ablation_results, metric='top5_acc',
                      save_path=f'{RESULTS_DIR}/ablation_bar_top5.png')

    # Print delta table
    full_top1 = ablation_results['Full model']['top1_acc']
    print('\n=== Accuracy drop vs full model (top-1) ===')
    for name, res in ablation_results.items():
        delta = res['top1_acc'] - full_top1
        print(f'  {name:<35s}  {res["top1_acc"]:.3f}  ({delta:+.3f})')


if __name__ == '__main__':
    main()
