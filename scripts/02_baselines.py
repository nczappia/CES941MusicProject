"""
Script 02 — Train and evaluate baselines

Run from project root:
    python scripts/02_baselines.py

Outputs:
    results/baseline_results.json
    results/baseline_comparison.png
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path

from src.dataset  import get_splits
from src.baselines import (
    MarkovBaseline, extract_sequences,
    train_lstm, evaluate_lstm,
)
from src.model    import LSTMBaseline
from src.visualize import plot_model_comparison

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED_DIR)

    train_c, train_s = extract_sequences(train_g)
    val_c,   val_s   = extract_sequences(val_g)
    test_c,  test_s  = extract_sequences(test_g)

    all_results = {}

    # ── Markov baseline ────────────────────────────────────────────────────
    print('\n=== Markov baseline ===')
    markov = MarkovBaseline()
    markov.fit(train_c, train_s)

    markov_val  = markov.evaluate(val_c,  val_s)
    markov_test = markov.evaluate(test_c, test_s)

    print(f'  Val   | top1={markov_val["top1_acc"]:.3f} '
          f'| top5={markov_val["top5_acc"]:.3f} '
          f'| top10={markov_val["top10_acc"]:.3f} '
          f'| CE={markov_val["cross_entropy"]:.4f}')
    print(f'  Test  | top1={markov_test["top1_acc"]:.3f} '
          f'| top5={markov_test["top5_acc"]:.3f} '
          f'| top10={markov_test["top10_acc"]:.3f} '
          f'| CE={markov_test["cross_entropy"]:.4f}')

    all_results['Markov'] = markov_test

    # ── LSTM baseline ──────────────────────────────────────────────────────
    print('\n=== LSTM baseline ===')
    lstm_model = LSTMBaseline(embed_dim=64, hidden_dim=256, num_layers=2, dropout=0.3)

    history = train_lstm(
        lstm_model,
        train_seqs=(train_c, train_s),
        val_seqs=(val_c, val_s),
        epochs=40,
        lr=1e-3,
        batch_size=32,
        device=DEVICE,
    )

    lstm_test = evaluate_lstm(lstm_model, test_c, test_s, device=DEVICE)
    print(f'\n  Test  | top1={lstm_test["top1_acc"]:.3f} '
          f'| top5={lstm_test["top5_acc"]:.3f} '
          f'| top10={lstm_test["top10_acc"]:.3f} '
          f'| CE={lstm_test["cross_entropy"]:.4f}')

    all_results['LSTM'] = lstm_test

    # Save LSTM model
    torch.save(lstm_model.state_dict(), f'{RESULTS_DIR}/lstm_baseline.pt')

    # ── Save results ───────────────────────────────────────────────────────
    with open(f'{RESULTS_DIR}/baseline_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved results to {RESULTS_DIR}/baseline_results.json')

    plot_model_comparison(
        all_results,
        metrics=['top1_acc', 'top5_acc', 'top10_acc'],
        save_path=f'{RESULTS_DIR}/baseline_comparison.png',
    )


if __name__ == '__main__':
    main()
