"""
Script 17 — Homogeneous GNN baseline.

Answers: "Does heterogeneous type-specialization in message passing actually
help, or is just having a GNN enough?"

Uses the identical causal edge set and chord-in-occ input enrichment as
causal v2 (MusicHeteroGNN, script 06), but replaces the per-relation
HeteroConv stack with a single SAGEConv that treats all edges identically.
All node types are projected to hidden_dim (unavoidable given different input
dims), and a learnable node-type embedding is added so the model knows what
kind of node it is — but the graph convolution itself is homogeneous.

Configuration: hidden_dim=128, num_layers=3, dropout=0.3, 60 epochs.

Outputs:
    results/homo_gnn_best.pt
    results/homo_gnn_history.json
    results/homo_gnn_training_curves.png
    results/homo_gnn_results.json

Run from project root:
    source venv/bin/activate && python scripts/17_homo_gnn.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset import get_splits
from src.model   import HomoMusicGNN
from src.train   import train_gnn, evaluate_gnn

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


def plot_curves(history, save_path):
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r['train_loss']          for r in history], label='train CE')
    axes[0].plot(epochs, [r['val_cross_entropy']   for r in history], label='val CE')
    axes[0].set_title('HomoGNN — cross-entropy')
    axes[0].set_xlabel('Epoch'); axes[0].legend()
    axes[1].plot(epochs, [r['val_top1_acc'] for r in history], label='top-1')
    axes[1].plot(epochs, [r['val_top5_acc'] for r in history], label='top-5')
    axes[1].set_title('HomoGNN — val accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved {save_path}')


def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED_DIR)

    model = HomoMusicGNN(hidden_dim=128, num_layers=3, dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'HomoMusicGNN parameters: {n_params:,}')

    print('\n=== Training Homogeneous GNN (60 epochs) ===')
    history = train_gnn(
        model, train_g, val_g,
        epochs=60, lr=1e-3, weight_decay=1e-4, batch_size=16,
        device=DEVICE,
        checkpoint_path=f'{RESULTS_DIR}/homo_gnn_best.pt',
    )

    model.load_state_dict(torch.load(f'{RESULTS_DIR}/homo_gnn_best.pt',
                                     map_location=DEVICE))
    test_metrics = evaluate_gnn(model, test_g, device=DEVICE)

    print(f'\n=== Test Results ===')
    print(f'  Top-1 acc:     {test_metrics["top1_acc"]:.4f}')
    print(f'  Top-5 acc:     {test_metrics["top5_acc"]:.4f}')
    print(f'  Top-10 acc:    {test_metrics["top10_acc"]:.4f}')
    print(f'  Cross-entropy: {test_metrics["cross_entropy"]:.4f}')

    with open(f'{RESULTS_DIR}/homo_gnn_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(f'{RESULTS_DIR}/homo_gnn_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    plot_curves(history, f'{RESULTS_DIR}/homo_gnn_training_curves.png')

    print(f'\nDone. Outputs in {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
