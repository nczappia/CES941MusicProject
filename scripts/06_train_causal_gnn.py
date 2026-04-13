"""
Script 06 — Train the causal HeteroGNN (no prev edges)

The full model includes occ→prev→occ edges, which create a 2-hop leakage path:
    chord[i+1] →(inst_rev)→ occ[i+1] →(prev)→ occ[i]
This lets occ[i] see chord[i+1]'s identity before prediction, explaining ~99.8% accuracy.

This script trains with use_prev_edges=False to measure honest next-chord prediction.

Run from project root:
    python scripts/06_train_causal_gnn.py

Outputs:
    results/causal_gnn_best.pt
    results/causal_gnn_history.json
    results/causal_gnn_test_results.json
    results/causal_gnn_training_curves.png
    results/causal_gnn_section_accuracy.png
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path

from src.dataset   import get_splits
from src.model     import MusicHeteroGNN
from src.train     import train_gnn, evaluate_gnn, evaluate_gnn_by_section
from src.visualize import plot_training_curves, plot_section_accuracy

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED_DIR)

    # ── Build causal model (no prev edges, chord features in occ) ────────────
    model = MusicHeteroGNN(
        hidden_dim=128,
        num_layers=3,           # 3 layers → 3-step backward receptive field via next edges
        dropout=0.3,
        use_prev_edges=False,   # causal: no future leakage
        use_chord_in_occ=True,  # inject current chord features into occ input (same signal as LSTM)
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Causal model parameters: {n_params:,}')
    print('  use_prev_edges=False  — occ[i] cannot see occ[i+1] features')
    print('  use_chord_in_occ=True — occ input includes current chord type features')
    print('  num_layers=3          — 3-step backward receptive field')

    # ── Train ──────────────────────────────────────────────────────────────
    print('\n=== Training Causal HeteroGNN ===')
    history = train_gnn(
        model,
        train_graphs=train_g,
        val_graphs=val_g,
        epochs=60,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        device=DEVICE,
        checkpoint_path=f'{RESULTS_DIR}/causal_gnn_best.pt',
    )

    # ── Load best checkpoint and evaluate on test ─────────────────────────
    model.load_state_dict(torch.load(f'{RESULTS_DIR}/causal_gnn_best.pt', map_location=DEVICE))
    test_metrics = evaluate_gnn(model, test_g, device=DEVICE)

    print(f'\n=== Causal GNN Test Results ===')
    for k, v in test_metrics.items():
        print(f'  {k}: {v:.4f}')

    # Per-section breakdown
    sec_results = evaluate_gnn_by_section(model, test_g, device=DEVICE)
    print(f'\n=== Per-section accuracy ===')
    for stype, vals in sorted(sec_results.items(), key=lambda x: -x[1]['count']):
        print(f'  {stype:<20s}  top1={vals["top1_acc"]:.3f}  (n={vals["count"]})')

    # ── Compare with full (leaky) model ───────────────────────────────────
    leaky_path = f'{RESULTS_DIR}/gnn_test_results.json'
    if os.path.exists(leaky_path):
        with open(leaky_path) as f:
            leaky = json.load(f)['test']
        print(f'\n=== Comparison ===')
        print(f'  {"Model":<25s}  Top-1    Top-5    CE')
        print(f'  {"Full (with prev)":<25s}  {leaky["top1_acc"]:.4f}   {leaky["top5_acc"]:.4f}   {leaky["cross_entropy"]:.4f}')
        print(f'  {"Causal (no prev)":<25s}  {test_metrics["top1_acc"]:.4f}   {test_metrics["top5_acc"]:.4f}   {test_metrics["cross_entropy"]:.4f}')

    # ── Save ───────────────────────────────────────────────────────────────
    with open(f'{RESULTS_DIR}/causal_gnn_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    with open(f'{RESULTS_DIR}/causal_gnn_test_results.json', 'w') as f:
        json.dump({'test': test_metrics, 'by_section': sec_results}, f, indent=2)

    plot_training_curves(history, save_path=f'{RESULTS_DIR}/causal_gnn_training_curves.png')
    plot_section_accuracy(sec_results, save_path=f'{RESULTS_DIR}/causal_gnn_section_accuracy.png')

    print(f'\nAll outputs saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
