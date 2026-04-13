"""
Visualization utilities.

    plot_training_curves(history, save_path)
    plot_embedding_umap(embeddings, labels, label_names, save_path)
    plot_ablation_bar(ablation_results, save_path)
    plot_section_accuracy(section_results, save_path)
    plot_top_chord_transitions(model, graphs, device, save_path)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Dict, Optional

from .vocab import SECTION_TYPES


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: List[Dict], save_path: str = None):
    epochs     = [r['epoch']      for r in history]
    train_loss = [r['train_loss'] for r in history]
    val_ce     = [r.get('val_cross_entropy', r.get('cross_entropy', 0)) for r in history]
    val_top1   = [r.get('val_top1_acc', r.get('top1_acc', 0)) for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_loss, label='train loss')
    ax1.plot(epochs, val_ce,     label='val CE')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training curves'); ax1.legend()

    ax2.plot(epochs, val_top1)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Top-1 accuracy')
    ax2.set_title('Validation accuracy')
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved {save_path}')
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# UMAP embedding plot
# ---------------------------------------------------------------------------

def plot_embedding_umap(
    embeddings:  torch.Tensor,
    labels:      torch.Tensor,
    label_names: List[str],
    title:       str = 'Song embedding (UMAP)',
    save_path:   str = None,
):
    try:
        import umap
    except ImportError:
        print('umap-learn not installed; skipping UMAP plot.')
        return

    X = embeddings.numpy()
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    Z = reducer.fit_transform(X)

    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=[cmap(i)], label=label_names[lab] if lab < len(label_names) else str(lab),
                   s=15, alpha=0.7)

    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8, loc='best')
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved {save_path}')
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation_bar(
    ablation_results: Dict[str, Dict],
    metric:     str = 'top1_acc',
    save_path:  str = None,
):
    """
    ablation_results: {'Full model': {metric: val}, 'No seq edges': {...}, ...}
    """
    names  = list(ablation_results.keys())
    values = [ablation_results[n].get(metric, 0) * 100 for n in names]   # as %

    colors = ['steelblue'] + ['#e07b54'] * (len(names) - 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color=colors, edgecolor='white', height=0.6)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)

    ax.set_xlabel(f'{metric.replace("_", " ").title()} (%)')
    ax.set_title('Ablation study — edge type contribution')
    ax.set_xlim(0, max(values) * 1.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved {save_path}')
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Per-section accuracy
# ---------------------------------------------------------------------------

def plot_section_accuracy(
    section_results: Dict[str, Dict],
    save_path: str = None,
):
    """section_results: section_type → {top1_acc, count}"""
    items  = sorted(section_results.items(), key=lambda x: -x[1]['count'])
    names  = [f'{k} (n={v["count"]})' for k, v in items]
    accs   = [v['top1_acc'] * 100 for _, v in items]

    fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.45)))
    ax.barh(names, accs, color='steelblue', edgecolor='white', height=0.6)
    for i, val in enumerate(accs):
        ax.text(val + 0.2, i, f'{val:.1f}%', va='center', fontsize=8)

    ax.set_xlabel('Top-1 accuracy (%)')
    ax.set_title('Per-section-type prediction accuracy')
    ax.set_xlim(0, max(accs) * 1.15 if accs else 100)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved {save_path}')
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = ('top1_acc', 'top5_acc', 'top10_acc'),
    save_path: str = None,
):
    """
    results: {'Markov': {metric: val}, 'LSTM': {...}, 'HeteroGNN': {...}}
    """
    model_names = list(results.keys())
    n_metrics   = len(metrics)
    x           = np.arange(len(model_names))
    width       = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, met in enumerate(metrics):
        vals = [results[m].get(met, 0) * 100 for m in model_names]
        ax.bar(x + i * width, vals, width, label=met.replace('_', ' ').title())

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model comparison — next-chord prediction')
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved {save_path}')
    else:
        plt.show()
    plt.close()
