"""
Script 16 — Causal Transformer baseline for next-chord prediction.

Answers: "Can a standard attention-based sequence model match the GNN?"

Architecture: 3-layer causal Transformer encoder (GPT-style) on chord ID
sequences.  Section type is added as a learned embedding alongside chord
and positional embeddings, matching the conditioning available to the LSTM.

Configuration: embed_dim=128, nhead=4, num_layers=3, dim_feedforward=512,
dropout=0.1, 60 epochs, batch_size=32, cosine LR schedule.

Outputs:
    results/transformer_best.pt
    results/transformer_history.json
    results/transformer_training_curves.png
    results/transformer_results.json

Run from project root:
    source venv/bin/activate && python scripts/16_transformer_baseline.py
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from src.dataset   import get_splits
from src.baselines import extract_sequences, collate_lstm, evaluate_lstm
from src.model     import TransformerBaseline, NUM_CLASSES

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS     = 60
LR         = 1e-3
BATCH_SIZE = 32

print(f'Using device: {DEVICE}')


# ── Training ──────────────────────────────────────────────────────────────────

def train_transformer(model, train_c, train_s, val_c, val_s):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_ce = float('inf')
    history     = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm    = list(range(len(train_c)))
        random.shuffle(perm)
        tc = [train_c[i] for i in perm]
        ts = [train_s[i] for i in perm]

        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(tc), BATCH_SIZE):
            batch = collate_lstm(tc[start:start+BATCH_SIZE],
                                 ts[start:start+BATCH_SIZE], device=DEVICE)
            if batch is None:
                continue
            chord_in, sec_in, targets, _ = batch

            optimizer.zero_grad()
            logits = model(chord_in, sec_in)          # [B, T, V]
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B*T, V), targets.view(B*T), ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        val_metrics = evaluate_lstm(model, val_c, val_s, device=DEVICE)
        row = {
            'epoch':      epoch,
            'train_loss': epoch_loss / max(n_batches, 1),
            **val_metrics,
        }
        history.append(row)

        if val_metrics['cross_entropy'] < best_val_ce:
            best_val_ce = val_metrics['cross_entropy']
            torch.save(model.state_dict(), f'{RESULTS_DIR}/transformer_best.pt')

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d} | train_loss={row["train_loss"]:.4f} '
                  f'| val_ce={val_metrics["cross_entropy"]:.4f} '
                  f'| val_top1={val_metrics["top1_acc"]:.3f} '
                  f'| val_top5={val_metrics["top5_acc"]:.3f}')

    return history


def plot_curves(history, save_path):
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r['train_loss']      for r in history], label='train CE')
    axes[0].plot(epochs, [r['cross_entropy']   for r in history], label='val CE')
    axes[0].set_title('Transformer — cross-entropy')
    axes[0].set_xlabel('Epoch'); axes[0].legend()
    axes[1].plot(epochs, [r['top1_acc'] for r in history], label='top-1')
    axes[1].plot(epochs, [r['top5_acc'] for r in history], label='top-5')
    axes[1].set_title('Transformer — val accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED_DIR)

    train_c, train_s = extract_sequences(train_g)
    val_c,   val_s   = extract_sequences(val_g)
    test_c,  test_s  = extract_sequences(test_g)

    model = TransformerBaseline(
        vocab_size=NUM_CLASSES,
        embed_dim=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        num_sections=11,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'TransformerBaseline parameters: {n_params:,}')

    print(f'\n=== Training Transformer (60 epochs) ===')
    history = train_transformer(model, train_c, train_s, val_c, val_s)

    model.load_state_dict(torch.load(f'{RESULTS_DIR}/transformer_best.pt',
                                     map_location=DEVICE))
    test_metrics = evaluate_lstm(model, test_c, test_s, device=DEVICE)

    print(f'\n=== Test Results ===')
    print(f'  Top-1 acc:     {test_metrics["top1_acc"]:.4f}')
    print(f'  Top-5 acc:     {test_metrics["top5_acc"]:.4f}')
    print(f'  Top-10 acc:    {test_metrics["top10_acc"]:.4f}')
    print(f'  Cross-entropy: {test_metrics["cross_entropy"]:.4f}')

    with open(f'{RESULTS_DIR}/transformer_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(f'{RESULTS_DIR}/transformer_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    plot_curves(history, f'{RESULTS_DIR}/transformer_training_curves.png')

    print(f'\nDone. Outputs in {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
