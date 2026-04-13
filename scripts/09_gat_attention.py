"""
Script 09 — Train causal GAT model and visualize attention weights

Replaces SAGEConv with GATConv (4 heads) in the causal heterogeneous GNN.
After training, extracts attention weights on the `next` edges to show
which prior chords the model attends to most when making predictions.

Key questions answered:
  - Does attention improve over mean aggregation (SAGEConv)?
  - Which step of chord history gets the most attention? (1 step ago vs 2 vs 3)
  - Do certain chord transitions attract stronger attention?

Run from project root:
    python scripts/09_gat_attention.py

Outputs:
    results/gat_best.pt
    results/gat_history.json
    results/gat_training_curves.png
    results/gat_attention_by_step.png     — avg attention weight by hop distance
    results/gat_attention_heatmap.png     — attention from chord X to chord Y
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from pathlib import Path

from src.dataset   import get_splits
from src.model     import MusicHeteroGNN, NUM_CLASSES
from src.train     import train_gnn, evaluate_gnn
from src.visualize import plot_training_curves
from src.vocab     import VOCAB_SIZE, N_CHORD_ID

DATA_DIR      = 'data/McGill-Billboard'
PROCESSED_DIR = 'data/processed'
RESULTS_DIR   = 'results'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ROOTS  = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
QUALS  = ['maj','min','dim','aug','sus','oth']

def chord_id_to_str(cid):
    if cid == N_CHORD_ID:
        return 'N'
    root = cid // 12
    qual = (cid % 12) // 2
    cx   = cid % 2
    return f'{ROOTS[root]}:{QUALS[qual]}{"7" if cx else ""}'


# ── Attention extraction ──────────────────────────────────────────────────

@torch.no_grad()
def extract_next_edge_attention(model, graph, device):
    """
    Run a single-song forward pass and extract attention weights on
    (occ, next, occ) edges from the first GATConv layer.

    Returns:
        edge_index : [2, E] — source/dest occ indices for 'next' edges
        attn_weights : [E] — mean attention weight across heads
        chord_ids : [N_occ] — chord vocab id per occ node
    """
    model.eval()
    graph = graph.to(device)

    from src.graph import CHORD_FEAT_DIM

    # Build occ input (same as forward pass)
    occ_input = graph['occ'].x
    if model.use_chord_in_occ:
        ei_inst_rev = graph.edge_index_dict.get(('chord', 'inst_rev', 'occ'))
        if ei_inst_rev is not None:
            chord_feat_per_occ = torch.zeros(
                occ_input.shape[0], CHORD_FEAT_DIM, device=device, dtype=occ_input.dtype)
            chord_feat_per_occ[ei_inst_rev[1]] = graph['chord'].x[ei_inst_rev[0]]
            occ_input = torch.cat([occ_input, chord_feat_per_occ], dim=1)

    x_sec = graph['sec'].x
    h = {
        'occ':   model.occ_proj(occ_input),
        'chord': model.chord_proj(graph['chord'].x),
        'sec':   model.sec_proj(x_sec),
    }

    # Get next-edge index
    ei_next = graph.edge_index_dict.get(('occ', 'next', 'occ'))
    if ei_next is None:
        return None, None, None

    # Run only layer 0, extract attention from the next-conv
    conv_layer = model.convs[0]  # HeteroConv
    _next_key  = ('occ', 'next', 'occ')
    if _next_key not in conv_layer.convs:
        return None, None, None
    next_conv = conv_layer.convs[_next_key]

    # GATConv with return_attention_weights
    _, (edge_idx, attn) = next_conv(
        (h['occ'], h['occ']),
        ei_next,
        return_attention_weights=True,
    )
    # attn: [E, heads] → mean over heads
    attn_mean = attn.mean(dim=1).cpu()

    # Recover chord id per occ from labels (y[i] = next chord, so current = y[i-1] roughly)
    # Better: use inst_rev to get the chord type this occ belongs to
    ei_inst_rev = graph.edge_index_dict.get(('chord', 'inst_rev', 'occ'))
    N_occ = occ_input.shape[0]
    chord_per_occ = torch.full((N_occ,), N_CHORD_ID, dtype=torch.long)
    if ei_inst_rev is not None:
        # inst_rev: chord_local_id → occ_id
        # We need: occ_id → global chord_id
        # chord node features encode the global chord id implicitly via one-hot
        # Recover from chord feature: root*12 + qual*2 + complexity
        chord_feats = graph['chord'].x  # [N_chord, 20]
        root_idx  = chord_feats[:, :12].argmax(dim=1)
        qual_idx  = chord_feats[:, 12:18].argmax(dim=1)
        cplx_idx  = chord_feats[:, 18:20].argmax(dim=1)
        cids_local = root_idx * 12 + qual_idx * 2 + cplx_idx  # [N_chord]
        chord_per_occ[ei_inst_rev[1].cpu()] = cids_local[ei_inst_rev[0].cpu()].cpu()

    return edge_idx.cpu(), attn_mean, chord_per_occ


# ── Plot 1: Average attention by hop distance ─────────────────────────────

def plot_attention_by_step(model, graphs, device, n_songs=200):
    """
    For each 'next' edge (occ[i] → occ[i+1]), compute the attention weight.
    Since next edges go i→i+1, the destination occ[i+1] attends BACK to occ[i].
    We compute the positional gap (always 1 for next edges in layer 1),
    but across layers the receptive field expands — so we measure per-layer.

    More interestingly: compare attention on next edges vs inst_rev edges
    to see whether the model relies more on sequence history or chord identity.
    """
    print('Extracting attention weights...')

    # Collect attention stats per relation type across songs
    relation_attn = defaultdict(list)

    for g in graphs[:n_songs]:
        g_dev = g.to(device)

        from src.graph import CHORD_FEAT_DIM
        occ_input = g_dev['occ'].x
        if model.use_chord_in_occ:
            ei_ir = g_dev.edge_index_dict.get(('chord', 'inst_rev', 'occ'))
            if ei_ir is not None:
                cfpo = torch.zeros(occ_input.shape[0], CHORD_FEAT_DIM,
                                   device=device, dtype=occ_input.dtype)
                cfpo[ei_ir[1]] = g_dev['chord'].x[ei_ir[0]]
                occ_input = torch.cat([occ_input, cfpo], dim=1)

        h = {
            'occ':   model.occ_proj(occ_input),
            'chord': model.chord_proj(g_dev['chord'].x),
            'sec':   model.sec_proj(g_dev['sec'].x),
        }

        conv_layer = model.convs[0]
        for rel_key, conv in conv_layer.convs.items():
            # rel_key is either a (src, rel, dst) tuple or 'src__rel__dst' string
            if isinstance(rel_key, tuple):
                if len(rel_key) != 3:
                    continue
                src_type, rel, dst_type = rel_key
            else:
                parts = rel_key.split('__')
                if len(parts) != 3:
                    continue
                src_type, rel, dst_type = parts
            edge_key = (src_type, rel, dst_type)
            ei = g_dev.edge_index_dict.get(edge_key)
            if ei is None or ei.shape[1] == 0:
                continue

            src_h = h[src_type]
            dst_h = h[dst_type]

            try:
                _, (_, attn) = conv(
                    (src_h, dst_h), ei, return_attention_weights=True)
                relation_attn[rel].append(attn.mean(dim=1).mean().item())
            except Exception:
                continue

    if not relation_attn:
        print('  No attention weights extracted.')
        return

    # Plot mean attention per relation type
    rels    = sorted(relation_attn.keys())
    means   = [np.mean(relation_attn[r]) for r in rels]
    stds    = [np.std(relation_attn[r])  for r in rels]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(rels)))
    bars = ax.bar(rels, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_ylabel('Mean attention weight (layer 1)')
    ax.set_title('Average GAT Attention Weight by Edge Relation Type\n'
                 '(higher = model relies more on this relation for aggregation)')
    ax.set_xticklabels(rels, rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    out = f'{RESULTS_DIR}/gat_attention_by_relation.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')


# ── Plot 2: Attention heatmap — which chord transitions get high attention ─

def plot_attention_transition_heatmap(model, graphs, device, n_songs=200, top_chords=12):
    """
    For next edges (occ[i] → occ[i+1]): the destination occ[i+1] attends to
    source occ[i]. Plot mean attention weight for each (chord[i], chord[i+1]) pair.
    High attention = model strongly uses the previous chord when predicting the next.
    """
    print('Building attention transition heatmap...')

    # attn_matrix[src_chord][dst_chord] = list of attention weights
    attn_by_transition = defaultdict(list)

    for g in graphs[:n_songs]:
        ei_next, attn_weights, chord_per_occ = extract_next_edge_attention(
            model, g, device)
        if ei_next is None:
            continue

        src_occs = ei_next[0]   # occ[i]   — the previous chord
        dst_occs = ei_next[1]   # occ[i+1] — the chord being predicted

        for k in range(src_occs.shape[0]):
            src_cid = chord_per_occ[src_occs[k]].item()
            dst_cid = chord_per_occ[dst_occs[k]].item()
            w       = attn_weights[k].item()
            if src_cid != N_CHORD_ID and dst_cid != N_CHORD_ID:
                attn_by_transition[(src_cid, dst_cid)].append(w)

    if not attn_by_transition:
        print('  No transitions found.')
        return

    # Find top_chords most frequent chord IDs
    chord_counts = defaultdict(int)
    for (s, d), vals in attn_by_transition.items():
        chord_counts[s] += len(vals)
        chord_counts[d] += len(vals)
    top_cids = [c for c, _ in sorted(chord_counts.items(), key=lambda x: -x[1])[:top_chords]]
    top_labels = [chord_id_to_str(c) for c in top_cids]

    # Build matrix
    mat = np.zeros((top_chords, top_chords))
    cnt = np.zeros((top_chords, top_chords))
    cid_to_idx = {c: i for i, c in enumerate(top_cids)}
    for (s, d), vals in attn_by_transition.items():
        if s in cid_to_idx and d in cid_to_idx:
            i, j = cid_to_idx[s], cid_to_idx[d]
            mat[i, j] += sum(vals)
            cnt[i, j] += len(vals)
    with np.errstate(invalid='ignore'):
        mat = np.where(cnt > 0, mat / cnt, 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(top_chords))
    ax.set_yticks(range(top_chords))
    ax.set_xticklabels(top_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(top_labels, fontsize=9)
    ax.set_xlabel('Destination chord (occ[i+1])')
    ax.set_ylabel('Source chord (occ[i])')
    ax.set_title('Mean GAT Attention Weight by Chord Transition\n'
                 '(occ[i] → occ[i+1]: how much occ[i+1] attends to occ[i])')
    plt.colorbar(im, ax=ax, label='Mean attention weight')
    plt.tight_layout()
    out = f'{RESULTS_DIR}/gat_attention_heatmap.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved {out}')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    train_g, val_g, test_g = get_splits(DATA_DIR, PROCESSED_DIR)

    # ── Build GAT model ───────────────────────────────────────────────────
    model = MusicHeteroGNN(
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        use_prev_edges=False,
        use_chord_in_occ=True,
        use_attention=True,
        gat_heads=4,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'GAT model parameters: {n_params:,}')
    print('  use_attention=True  — GATConv (4 heads) instead of SAGEConv')
    print('  use_prev_edges=False — causal, no leakage')

    # ── Train ─────────────────────────────────────────────────────────────
    print('\n=== Training Causal GAT HeteroGNN ===')
    history = train_gnn(
        model,
        train_graphs=train_g,
        val_graphs=val_g,
        epochs=60,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        device=DEVICE,
        checkpoint_path=f'{RESULTS_DIR}/gat_best.pt',
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(f'{RESULTS_DIR}/gat_best.pt', map_location=DEVICE))
    test_metrics = evaluate_gnn(model, test_g, device=DEVICE)

    print(f'\n=== GAT Test Results ===')
    for k, v in test_metrics.items():
        print(f'  {k}: {v:.4f}')

    # Compare with SAGEConv causal model
    causal_path = f'{RESULTS_DIR}/causal_gnn_test_results.json'
    if os.path.exists(causal_path):
        with open(causal_path) as f:
            causal = json.load(f)['test']
        print(f'\n=== Comparison: SAGEConv vs GATConv ===')
        print(f'  {"Model":<30s}  Top-1    Top-5    CE')
        print(f'  {"Causal SAGEConv (v2)":<30s}  {causal["top1_acc"]:.4f}   {causal["top5_acc"]:.4f}   {causal["cross_entropy"]:.4f}')
        print(f'  {"Causal GATConv (4 heads)":<30s}  {test_metrics["top1_acc"]:.4f}   {test_metrics["top5_acc"]:.4f}   {test_metrics["cross_entropy"]:.4f}')

    # ── Save training artifacts ────────────────────────────────────────────
    with open(f'{RESULTS_DIR}/gat_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, save_path=f'{RESULTS_DIR}/gat_training_curves.png')

    # ── Attention visualizations ──────────────────────────────────────────
    model = model.to(DEVICE)
    model.eval()
    all_graphs = train_g + val_g + test_g

    print('\n=== Extracting Attention Weights ===')
    plot_attention_by_step(model, all_graphs, DEVICE)
    plot_attention_transition_heatmap(model, all_graphs, DEVICE)

    print(f'\nAll outputs saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
