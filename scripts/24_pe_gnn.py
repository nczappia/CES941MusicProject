"""
Script 24 — Position-Aware Heterogeneous GNN.

Addresses GNN permutation invariance: standard SAGEConv has no notion of whether
chord occurrence i comes before or after occurrence j within a section. This is
partially handled by the `next` edge, but that only covers direct neighbours.

Fix: inject sinusoidal positional encoding (PE_DIM=16) into occ node features,
based on each occ's normalised rank within its section (0 = first chord, 1 = last).
The PE is computed from the existing in_section edge index — no graph rebuild needed.

Architecture: identical to Causal GNN v2 (no prev, chord-in-occ, 3 layers, hidden=128).
Only change is occ.x grows from 19→35 dims (occ_proj adjusted accordingly).

Outputs:
  results/pe_gnn_best.pt
  results/pe_gnn_history.json
  results/pe_gnn_training_curves.png
  results/pe_gnn_test_results.json

Run from project root:
    source venv/bin/activate && python scripts/24_pe_gnn.py
"""

import sys, os, json, math, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.model import MusicHeteroGNN, NUM_CLASSES
from src.graph import OCC_FEAT_DIM, CHORD_FEAT_DIM
from src.dataset import get_splits
from src.train import train_gnn, evaluate_gnn

RESULTS   = 'results'
PROCESSED = 'data/processed'
DATA_DIR  = 'data/McGill-Billboard'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
PE_DIM    = 16


# ── Sinusoidal PE injection ───────────────────────────────────────────────────

def sinusoidal_pe(positions: torch.Tensor, pe_dim: int) -> torch.Tensor:
    """
    positions : [N] float in [0, 1] — normalised within-section rank
    Returns   : [N, pe_dim] sinusoidal encoding
    """
    pe = torch.zeros(len(positions), pe_dim)
    div = torch.exp(
        torch.arange(0, pe_dim, 2).float() * (-math.log(10000.0) / pe_dim)
    )
    pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div)
    pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div)
    return pe


def add_positional_encoding(graphs, pe_dim: int = PE_DIM):
    """
    Append pe_dim sinusoidal dims to occ.x in-place for every graph.
    Position is normalised rank within the occ's section (0 = first, 1 = last).
    """
    et = ('occ', 'in_section', 'sec')
    for g in graphs:
        N_occ   = g['occ'].x.shape[0]
        ei      = g[et].edge_index   # [2, N_occ]: row0=occ_idx, row1=sec_idx

        # Group occ indices by section; they arrive in increasing occ order
        sec_to_occs = defaultdict(list)
        for occ_idx, sec_idx in zip(ei[0].tolist(), ei[1].tolist()):
            sec_to_occs[sec_idx].append(occ_idx)

        pos = torch.zeros(N_occ)
        for occ_list in sec_to_occs.values():
            n = len(occ_list)
            for rank, occ_idx in enumerate(occ_list):
                pos[occ_idx] = rank / max(n - 1, 1)

        pe = sinusoidal_pe(pos, pe_dim)
        g['occ'].x = torch.cat([g['occ'].x, pe], dim=1)


# ── Model: swap occ_proj for the wider input dim ──────────────────────────────

class PEMusicHeteroGNN(MusicHeteroGNN):
    """
    Identical to MusicHeteroGNN causal v2 but with a larger occ_proj to
    accommodate the extra PE dimensions appended to occ.x.
    """
    def __init__(self, pe_dim: int = PE_DIM, **kwargs):
        super().__init__(**kwargs)
        use_chord = kwargs.get('use_chord_in_occ', True)
        # occ.x is now OCC_FEAT_DIM + pe_dim; chord injection adds CHORD_FEAT_DIM inside _encode
        occ_in_dim = (OCC_FEAT_DIM + pe_dim) + (CHORD_FEAT_DIM if use_chord else 0)
        hidden_dim = kwargs.get('hidden_dim', 128)
        self.occ_proj = nn.Linear(occ_in_dim, hidden_dim)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    Path(RESULTS).mkdir(exist_ok=True)
    print(f'Device: {DEVICE}')
    print(f'PE dim: {PE_DIM}  →  occ.x: {OCC_FEAT_DIM} → {OCC_FEAT_DIM + PE_DIM} dims')

    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED)
    print(f'Graphs: {len(train_g)} train / {len(val_g)} val / {len(test_g)} test')

    print('Injecting positional encoding into occ features...')
    add_positional_encoding(train_g)
    add_positional_encoding(val_g)
    add_positional_encoding(test_g)
    print(f'  occ.x shape after PE: {train_g[0]["occ"].x.shape}')

    model = PEMusicHeteroGNN(
        pe_dim          = PE_DIM,
        hidden_dim      = 128,
        num_layers      = 3,
        dropout         = 0.3,
        use_prev_edges  = False,
        use_chord_in_occ= True,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    print('\n=== Training Position-Aware GNN ===')
    history = train_gnn(
        model,
        train_g, val_g,
        epochs          = 60,
        lr              = 1e-3,
        weight_decay    = 1e-4,
        batch_size      = 16,
        device          = DEVICE,
        checkpoint_path = f'{RESULTS}/pe_gnn_best.pt',
    )

    with open(f'{RESULTS}/pe_gnn_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Training curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs  = [r['epoch']      for r in history]
    tr_loss = [r['train_loss'] for r in history]
    val_ce  = [r['val_cross_entropy'] for r in history]
    val_t1  = [r['val_top1_acc']      for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor='#0f0f1a')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0f0f1a'); ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#444')

    ax1.plot(epochs, tr_loss, color='#4CAF50', label='train loss')
    ax1.plot(epochs, val_ce,  color='#FF7F0E', label='val CE', linestyle='--')
    ax1.set_xlabel('Epoch', color='white'); ax1.set_ylabel('Cross-Entropy', color='white')
    ax1.set_title('Loss', color='white'); ax1.legend(labelcolor='white', framealpha=0.3)

    ax2.plot(epochs, val_t1, color='#2196F3', label='val top-1')
    ax2.set_xlabel('Epoch', color='white'); ax2.set_ylabel('Accuracy', color='white')
    ax2.set_title('Top-1 Accuracy', color='white'); ax2.legend(labelcolor='white', framealpha=0.3)

    fig.suptitle('Position-Aware GNN Training', color='white', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{RESULTS}/pe_gnn_training_curves.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f'  Saved {RESULTS}/pe_gnn_training_curves.png')

    # Load best and evaluate on test set
    model.load_state_dict(torch.load(f'{RESULTS}/pe_gnn_best.pt', map_location=DEVICE))
    model.to(DEVICE)

    print('\n=== Test Evaluation ===')
    test_metrics = evaluate_gnn(model, test_g, device=DEVICE)
    print(f'  Top-1 : {test_metrics["top1_acc"]:.3f}')
    print(f'  Top-5 : {test_metrics["top5_acc"]:.3f}')
    print(f'  CE    : {test_metrics["cross_entropy"]:.4f}')

    results = {
        'test': test_metrics,
        'baselines': {
            'markov':       0.235,
            'lstm':         0.452,
            'causal_gnn_v1': 0.460,
            'causal_gnn_v2': 0.608,
            'gat':          0.613,
        },
    }
    with open(f'{RESULTS}/pe_gnn_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nDone.')
    print(f'  PE GNN top-1: {test_metrics["top1_acc"]:.1%}')
    print(f'  Causal v2:    60.8%  |  GAT: 61.3%')


if __name__ == '__main__':
    main()
