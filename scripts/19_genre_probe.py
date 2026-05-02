"""
Script 19 — Genre classification probe for Transformer and HomoGNN.

Loads pretrained checkpoints (no retraining), extracts song-level embeddings,
trains a linear genre head on train embeddings, evaluates on test.

This is a standard linear probe — it measures how much genre information is
already encoded in the model's representations without any genre supervision.
Comparable to the full multitask results (scripts 10, 15) but without the
advantage of joint training; differences highlight what genre-supervised
fine-tuning adds.

Outputs:
    results/transformer_genre_acc.json
    results/homo_gnn_genre_acc.json
    results/genre_probe_comparison.png  — bar chart across all genre models

Run from project root:
    source venv/bin/activate && python scripts/19_genre_probe.py
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

from src.dataset   import get_splits
from src.baselines import extract_sequences, collate_lstm
from src.model     import TransformerBaseline, HomoMusicGNN, NUM_CLASSES

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'
GENRE_PATH    = 'data/genre_labels.json'

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_GENRE_COUNT = 10
PROBE_EPOCHS    = 40
PROBE_LR        = 1e-3
print(f'Using device: {DEVICE}')


# ── Genre helpers (same as scripts 10, 15) ────────────────────────────────────

def build_genre_vocab(genre_json):
    counts   = Counter(v['coarse'] for v in genre_json.values() if v.get('coarse'))
    valid    = {g for g, c in counts.items() if c >= MIN_GENRE_COUNT}
    genres   = sorted(valid - {'other'}) + ['other']
    genre2id = {g: i for i, g in enumerate(genres)}
    song2genre = {}
    for sid, info in genre_json.items():
        coarse = info.get('coarse')
        if coarse is None:
            continue
        if coarse not in valid:
            coarse = 'other'
        song2genre[sid] = genre2id[coarse]
    label_counts = Counter(song2genre.values())
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (len(genres) * label_counts[i]) for i in range(len(genres))],
        dtype=torch.float,
    )
    return song2genre, genre2id, genres, weights


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_transformer_embeddings(model, graphs, song2genre):
    """Mean-pool final hidden states over sequence length → one vec per song."""
    model.eval()
    embs, labels = [], []
    for g in graphs:
        sid = getattr(g, 'song_id', '')
        label = song2genre.get(sid, -1)

        # Reconstruct chord + section sequences (same logic as extract_sequences)
        ei = g['occ', 'instance_of', 'chord'].edge_index
        chord_local = ei[1]
        chord_feats = g['chord'].x
        root_ids  = chord_feats[:, :12].argmax(dim=1)
        qual_ids  = chord_feats[:, 12:18].argmax(dim=1)
        cplx_ids  = chord_feats[:, 18:20].argmax(dim=1)
        global_ids = root_ids * 12 + qual_ids * 2 + cplx_ids
        occ_order  = ei[0].argsort()
        chord_seq  = global_ids[chord_local[occ_order]]

        ei_secrev   = g['sec', 'sec_rev', 'occ'].edge_index
        sec_for_occ = torch.zeros(chord_seq.shape[0], dtype=torch.long)
        sec_for_occ[ei_secrev[1]] = ei_secrev[0]
        sec_type_ids = g['sec'].x[:, 2:].argmax(dim=1)
        sec_seq = sec_type_ids[sec_for_occ]

        if len(chord_seq) < 2:
            continue

        chord_in = chord_seq.unsqueeze(0).to(DEVICE)   # [1, T]
        sec_in   = sec_seq.unsqueeze(0).to(DEVICE)
        emb = model.encode_song(chord_in, sec_in).squeeze(0).cpu()  # [embed_dim]
        embs.append(emb)
        labels.append(label)

    return torch.stack(embs), labels


@torch.no_grad()
def extract_homo_embeddings(model, graphs, song2genre):
    """Mean-pool occ embeddings per song."""
    model.eval()
    embs, labels = [], []
    for g in graphs:
        sid   = getattr(g, 'song_id', '')
        label = song2genre.get(sid, -1)
        g2    = g.to(DEVICE)
        occ_emb = model.encode_occ(g2.x_dict, g2.edge_index_dict)  # [N_occ, H]
        embs.append(occ_emb.mean(dim=0).cpu())
        labels.append(label)
    return torch.stack(embs), labels


# ── Linear probe ─────────────────────────────────────────────────────────────

def train_linear_probe(train_embs, train_labels, num_genres, class_weights, device):
    """Train a single linear layer on frozen embeddings."""
    head = nn.Linear(train_embs.shape[1], num_genres).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    wt   = class_weights.to(device)

    # Filter to labelled only
    mask = [i for i, l in enumerate(train_labels) if l != -1]
    embs_k  = train_embs[mask].to(device)
    labs_k  = torch.tensor([train_labels[i] for i in mask], dtype=torch.long, device=device)

    for _ in range(PROBE_EPOCHS):
        head.train()
        perm = torch.randperm(embs_k.shape[0])
        for start in range(0, embs_k.shape[0], 64):
            idx = perm[start:start+64]
            opt.zero_grad()
            loss = F.cross_entropy(head(embs_k[idx]), labs_k[idx], weight=wt)
            loss.backward()
            opt.step()

    return head


@torch.no_grad()
def eval_probe(head, embs, labels, id2genre, device):
    head.eval()
    mask = [i for i, l in enumerate(labels) if l != -1]
    if not mask:
        return 0.0, {}
    embs_k = embs[mask].to(device)
    labs_k = torch.tensor([labels[i] for i in mask], dtype=torch.long, device=device)
    preds  = head(embs_k).argmax(dim=1)
    overall = (preds == labs_k).float().mean().item()

    per_class = {}
    for gid, gname in enumerate(id2genre):
        sel = labs_k == gid
        if sel.any():
            per_class[gname] = {
                'acc': (preds[sel] == labs_k[sel]).float().mean().item(),
                'n':   sel.sum().item(),
            }
    return overall, per_class


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_comparison(results, save_path):
    """Bar chart: overall genre accuracy across all genre-capable models."""
    models  = list(results.keys())
    accs    = [results[m]['overall'] * 100 for m in models]
    colors  = ['#ff7f0e', '#ff7f0e', '#2ca02c', '#2ca02c', '#1f77b4', '#1f77b4'][:len(models)]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(models, accs, color=colors, alpha=0.88, width=0.55, zorder=3)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Genre Top-1 Accuracy (%)', fontsize=11)
    ax.set_title('Genre Classification — All Models (Test Set)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 65)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    patches = [
        mpatches.Patch(color='#ff7f0e', alpha=0.88, label='Probe (linear head on frozen repr.)'),
        mpatches.Patch(color='#2ca02c', alpha=0.88, label='Joint training (chord + genre loss)'),
        mpatches.Patch(color='#1f77b4', alpha=0.88, label='HGT (hierarchical sec pooling)'),
    ]
    ax.legend(handles=patches, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    with open(GENRE_PATH) as f:
        genre_json = json.load(f)
    song2genre, genre2id, id2genre, class_weights = build_genre_vocab(genre_json)
    num_genres = len(id2genre)
    print(f'Genre classes ({num_genres}): {id2genre}')

    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED_DIR)
    all_splits = {'train': train_g, 'val': val_g, 'test': test_g}

    genre_results = {}

    # ── Transformer probe ─────────────────────────────────────────────────────
    print('\n=== Transformer genre probe ===')
    transformer = TransformerBaseline(
        vocab_size=NUM_CLASSES, embed_dim=128, nhead=4, num_layers=3,
        dim_feedforward=512, dropout=0.1, num_sections=11,
    ).to(DEVICE)
    transformer.load_state_dict(
        torch.load(f'{RESULTS_DIR}/transformer_best.pt', map_location=DEVICE)
    )

    print('  Extracting embeddings...')
    tr_embs, tr_labs = extract_transformer_embeddings(transformer, train_g, song2genre)
    te_embs, te_labs = extract_transformer_embeddings(transformer, test_g,  song2genre)

    print(f'  Training linear probe ({PROBE_EPOCHS} epochs)...')
    head = train_linear_probe(tr_embs, tr_labs, num_genres, class_weights, DEVICE)
    overall, per_class = eval_probe(head, te_embs, te_labs, id2genre, DEVICE)
    print(f'  Genre accuracy: {overall:.4f}')
    for gname, d in sorted(per_class.items(), key=lambda x: -x[1]['n']):
        print(f'    {gname:<15s}  acc={d["acc"]:.3f}  n={d["n"]}')

    transformer_res = {'overall': overall, 'per_class': per_class}
    with open(f'{RESULTS_DIR}/transformer_genre_acc.json', 'w') as f:
        json.dump(transformer_res, f, indent=2)
    genre_results['Transformer\n(probe)'] = transformer_res

    # ── HomoGNN probe ─────────────────────────────────────────────────────────
    print('\n=== HomoGNN genre probe ===')
    homo = HomoMusicGNN(hidden_dim=128, num_layers=3, dropout=0.3).to(DEVICE)
    homo.load_state_dict(
        torch.load(f'{RESULTS_DIR}/homo_gnn_best.pt', map_location=DEVICE)
    )

    print('  Extracting embeddings...')
    tr_embs, tr_labs = extract_homo_embeddings(homo, train_g, song2genre)
    te_embs, te_labs = extract_homo_embeddings(homo, test_g,  song2genre)

    print(f'  Training linear probe ({PROBE_EPOCHS} epochs)...')
    head = train_linear_probe(tr_embs, tr_labs, num_genres, class_weights, DEVICE)
    overall, per_class = eval_probe(head, te_embs, te_labs, id2genre, DEVICE)
    print(f'  Genre accuracy: {overall:.4f}')
    for gname, d in sorted(per_class.items(), key=lambda x: -x[1]['n']):
        print(f'    {gname:<15s}  acc={d["acc"]:.3f}  n={d["n"]}')

    homo_res = {'overall': overall, 'per_class': per_class}
    with open(f'{RESULTS_DIR}/homo_gnn_genre_acc.json', 'w') as f:
        json.dump(homo_res, f, indent=2)
    genre_results['HomoGNN\n(probe)'] = homo_res

    # ── Add existing results for comparison ───────────────────────────────────
    with open(f'{RESULTS_DIR}/multitask_genre_acc.json') as f:
        mt = json.load(f)
    genre_results['Het. GNN\n(multitask)'] = {
        'overall': mt['test_genre_overall'],
        'per_class': mt['test_genre_per_class'],
    }

    with open(f'{RESULTS_DIR}/hgt_genre_acc.json') as f:
        hgt = json.load(f)
    genre_results['HGT\n(hierarchical)'] = {
        'overall': hgt['test_genre_overall'],
        'per_class': hgt['test_genre_per_class'],
    }

    plot_comparison(genre_results, f'{RESULTS_DIR}/genre_probe_comparison.png')

    print('\n=== Genre Summary ===')
    for name, res in genre_results.items():
        print(f'  {name.replace(chr(10), " "):<30s}  {res["overall"]*100:.1f}%')

    print(f'\nDone. Outputs in {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
