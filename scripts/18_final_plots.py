"""
Script 18 — Final presentation plots.

Generates:
    results/model_comparison_final.png  — horizontal bar chart, all models
    results/design_ladder.png           — incremental gain from each design choice
    results/ablation_final.png          — clean ablation bar (causal model variants)

Run from project root:
    source venv/bin/activate && python scripts/18_final_plots.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_DIR = 'results'
Path(RESULTS_DIR).mkdir(exist_ok=True)

# ── All results (test set) ────────────────────────────────────────────────────

MODELS = [
    # (label,            top1,   top5,   ce,     family)
    ('Markov',           0.2350, 0.6057, 4.577,  'Sequence Baselines'),
    ('LSTM',             0.4516, 0.8313, 1.975,  'Sequence Baselines'),
    ('Transformer',      0.5220, 0.8458, 1.842,  'Sequence Baselines'),
    ('Homo GNN',         0.5572, 0.8684, 1.597,  'Graph Baselines'),
    ('HGT',              0.5551, 0.8397, 1.759,  'Graph Baselines'),
    ('Multi-task GNN',   0.5872, 0.8775, 1.520,  'Ours'),
    ('GAT (causal)',     0.6030, 0.9095, 1.440,  'Ours'),
    ('Het. GNN (ours)',  0.6075, 0.8901, 1.443,  'Ours'),
]

FAMILY_COLORS = {
    'Sequence Baselines': '#d62728',   # red
    'Graph Baselines':    '#ff7f0e',   # orange
    'Ours':               '#2ca02c',   # green
}

# ── 1. Full model comparison bar chart ───────────────────────────────────────

def plot_model_comparison():
    labels  = [m[0] for m in MODELS]
    top1    = [m[1] for m in MODELS]
    top5    = [m[2] for m in MODELS]
    colors  = [FAMILY_COLORS[m[4]] for m in MODELS]

    x      = np.arange(len(labels))
    width  = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))

    bars1 = ax.bar(x - width/2, [v*100 for v in top1], width,
                   color=colors, alpha=0.9, label='Top-1', zorder=3)
    bars2 = ax.bar(x + width/2, [v*100 for v in top5], width,
                   color=colors, alpha=0.45, label='Top-5', zorder=3,
                   hatch='//')

    # Value labels on top-1 bars
    for bar, val in zip(bars1, top1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val*100:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Next-Chord Prediction — All Models (Test Set)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Family legend patches
    family_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
    bar_patches    = [
        mpatches.Patch(facecolor='grey', alpha=0.9,  label='Top-1'),
        mpatches.Patch(facecolor='grey', alpha=0.45, hatch='//', label='Top-5'),
    ]
    ax.legend(handles=family_patches + bar_patches, fontsize=9,
              loc='upper left', ncol=2)

    fig.tight_layout()
    path = f'{RESULTS_DIR}/model_comparison_final.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ── 2. Design-choice ladder (incremental gains) ───────────────────────────────

def plot_design_ladder():
    """
    Shows how each design decision adds top-1 accuracy over the previous step.
    Stacked horizontal bar — each segment = gain from one decision.
    """
    steps = [
        ('Markov\n(bigram)',         0.2350, '#d62728'),
        ('+Sequential\nmodeling\n(LSTM)',       0.4516 - 0.2350, '#e07070'),
        ('+Self-attention\n(Transformer)',       0.5220 - 0.4516, '#ff7f0e'),
        ('+Graph\nstructure\n(Homo GNN)',        0.5572 - 0.5220, '#ffa940'),
        ('+Type\nspecialization\n(Het. GNN)',    0.6075 - 0.5572, '#2ca02c'),
    ]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    left = 0.0
    for label, width, color in steps:
        ax.barh(0, width*100, left=left*100, color=color, edgecolor='white',
                linewidth=1.5, height=0.55)
        cx = (left + width/2) * 100
        sign = '+' if label != steps[0][0] else ''
        pct  = f'{sign}{width*100:.1f}%'
        ax.text(cx, 0, f'{pct}\n{label}',
                ha='center', va='center', fontsize=8.5, fontweight='bold',
                color='white' if width > 0.04 else 'black', linespacing=1.4)
        left += width

    ax.set_xlim(0, 70)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel('Top-1 Accuracy (%)', fontsize=11)
    ax.set_title('Incremental Gain from Each Design Decision', fontsize=13, fontweight='bold')
    ax.set_yticks([])
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Final accuracy annotation
    ax.axvline(left*100, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(left*100 + 0.5, 0.35, f'Final: {left*100:.1f}%', color='#2ca02c',
            fontsize=10, fontweight='bold')

    fig.tight_layout()
    path = f'{RESULTS_DIR}/design_ladder.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ── 3. Clean ablation bar ─────────────────────────────────────────────────────

def plot_ablation_clean():
    """
    Ablation over the causal/hetero GNN family showing what each component contributes.
    Uses existing ablation_results.json data plus homo/HGT for context.
    """
    variants = [
        # (label,                      top1,   color)
        ('No seq edges\n(structure only)', 0.2536, '#d62728'),
        ('No instance_of\n(no chord→occ)',  0.0752, '#d62728'),
        ('Homo GNN\n(flat conv)',           0.5572, '#ff7f0e'),
        ('HGT\n(hetero attn)',              0.5551, '#ff7f0e'),
        ('Het. GNN v2\n(ours)',             0.6075, '#2ca02c'),
        ('GAT\n(ours+attn)',                0.6030, '#2ca02c'),
    ]

    labels = [v[0] for v in variants]
    vals   = [v[1]*100 for v in variants]
    colors = [v[2] for v in variants]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.88, zorder=3, width=0.6)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Reference line: our best
    ax.axhline(60.75, color='#2ca02c', linestyle='--', linewidth=1.4, alpha=0.7, zorder=2)

    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
    ax.set_title('Ablation: GNN Architecture Choices (Test Set)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 72)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(color='#d62728', alpha=0.88, label='Ablated / broken'),
        mpatches.Patch(color='#ff7f0e', alpha=0.88, label='Baseline graphs'),
        mpatches.Patch(color='#2ca02c', alpha=0.88, label='Our models'),
    ]
    ax.legend(handles=patches, fontsize=9)

    fig.tight_layout()
    path = f'{RESULTS_DIR}/ablation_final.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ── 4. Results table as figure ───────────────────────────────────────────────

def plot_results_table():
    rows = [
        ['Markov',          '23.5%', '60.6%', '4.58',  'Sequence'],
        ['LSTM',            '45.2%', '83.1%', '1.98',  'Sequence'],
        ['Transformer',     '52.2%', '84.6%', '1.84',  'Sequence'],
        ['Homo GNN',        '55.7%', '86.8%', '1.60',  'Graph'],
        ['HGT',             '55.5%', '84.0%', '1.76',  'Graph'],
        ['Multi-task GNN',  '58.7%', '87.7%', '1.52',  'Ours'],
        ['GAT (causal)',    '60.3%', '90.9%', '1.44',  'Ours'],
        ['Het. GNN (ours)', '60.7%', '89.0%', '1.44',  'Ours'],
    ]
    col_labels = ['Model', 'Top-1', 'Top-5', 'CE', 'Family']
    row_colors = []
    for r in rows:
        fam = r[-1]
        if fam == 'Sequence':
            row_colors.append(['#fdd5d5']*5)
        elif fam == 'Graph':
            row_colors.append(['#ffe4cc']*5)
        else:
            row_colors.append(['#d5f5d5']*5)
    # Highlight best row
    row_colors[-2] = ['#b8e8b8']*5   # GAT row
    row_colors[-1] = ['#a8dfa8']*5   # Het GNN best

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=row_colors,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    # Bold header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#333333')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Next-Chord Prediction — All Results (Test Set)',
                 fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    path = f'{RESULTS_DIR}/results_table.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ── 5. Genre results table as figure ─────────────────────────────────────────

def plot_genre_table():
    rows = [
        ['Transformer',      '23.9%', 'Linear probe on frozen repr.',  'Sequence'],
        ['Homo GNN',         '23.9%', 'Linear probe on frozen repr.',  'Graph'],
        ['Het. GNN (SAGE)',  '35.8%', 'Joint chord + genre training',  'Ours'],
        ['HGT',              '50.7%', 'Joint + hierarchical sec pool', 'Ours'],
    ]
    col_labels = ['Model', 'Genre Acc', 'Method', 'Family']
    row_colors = [
        ['#fdd5d5']*4,
        ['#ffe4cc']*4,
        ['#d5f5d5']*4,
        ['#a8dfa8']*4,
    ]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=row_colors,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#333333')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('Genre Classification — All Models (Test Set)',
                 fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    path = f'{RESULTS_DIR}/genre_table.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


if __name__ == '__main__':
    plot_model_comparison()
    plot_design_ladder()
    plot_ablation_clean()
    plot_results_table()
    plot_genre_table()
    print('\nAll plots saved.')
