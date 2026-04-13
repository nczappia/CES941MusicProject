"""
Script 03 — Train the full HeteroGNN

Run from project root:
    python scripts/03_train_gnn.py

Outputs:
    results/gnn_best.pt          (best checkpoint by val loss)
    results/gnn_history.json
    results/gnn_training_curves.png
    results/gnn_test_results.json
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

    # ── Build model ────────────────────────────────────────────────────────
    model = MusicHeteroGNN(
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    # ── Train ──────────────────────────────────────────────────────────────
    print('\n=== Training HeteroGNN ===')
    history = train_gnn(
        model,
        train_graphs=train_g,
        val_graphs=val_g,
        epochs=60,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        device=DEVICE,
        checkpoint_path=f'{RESULTS_DIR}/gnn_best.pt',
    )

    # ── Load best checkpoint and evaluate on test ─────────────────────────
    model.load_state_dict(torch.load(f'{RESULTS_DIR}/gnn_best.pt', map_location=DEVICE))
    test_metrics = evaluate_gnn(model, test_g, device=DEVICE)

    print(f'\n=== Test results ===')
    for k, v in test_metrics.items():
        print(f'  {k}: {v:.4f}')

    # Per-section breakdown
    sec_results = evaluate_gnn_by_section(model, test_g, device=DEVICE)
    print(f'\n=== Per-section accuracy ===')
    for stype, vals in sorted(sec_results.items(), key=lambda x: -x[1]['count']):
        print(f'  {stype:<20s}  top1={vals["top1_acc"]:.3f}  (n={vals["count"]})')

    # ── Save ───────────────────────────────────────────────────────────────
    with open(f'{RESULTS_DIR}/gnn_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    with open(f'{RESULTS_DIR}/gnn_test_results.json', 'w') as f:
        json.dump({'test': test_metrics, 'by_section': sec_results}, f, indent=2)

    plot_training_curves(history, save_path=f'{RESULTS_DIR}/gnn_training_curves.png')
    plot_section_accuracy(sec_results, save_path=f'{RESULTS_DIR}/gnn_section_accuracy.png')

    print(f'\nAll outputs saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
