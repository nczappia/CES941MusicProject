"""
Training and evaluation utilities for MusicHeteroGNN.

Key functions
─────────────
    train_gnn(model, train_graphs, val_graphs, ...)
    evaluate_gnn(model, graphs, ...)       — returns accuracy metrics
    evaluate_gnn_by_section(model, graphs) — per-section-type breakdown
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from typing import List, Dict, Optional, Tuple
import random

from .model import MusicHeteroGNN, NUM_CLASSES
from .vocab import VOCAB_SIZE, section_type_to_id, SECTION_TYPES


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def make_batch(graphs: List[HeteroData], device: str) -> HeteroData:
    """Collate a list of HeteroData into a single batched graph."""
    return Batch.from_data_list(graphs).to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gnn(
    model:       MusicHeteroGNN,
    train_graphs: List[HeteroData],
    val_graphs:   List[HeteroData],
    epochs:       int = 50,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size:   int = 16,
    device:       str = 'cuda',
    checkpoint_path: Optional[str] = None,
) -> List[Dict]:
    """Train MusicHeteroGNN; returns per-epoch history list."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_graphs)

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[start:start + batch_size]
            batch = make_batch(batch_graphs, device)

            optimizer.zero_grad()
            logits = model(batch.x_dict, batch.edge_index_dict)  # [N_occ, C]
            labels = batch['occ'].y                               # [N_occ]

            mask  = labels != -100
            loss  = F.cross_entropy(logits[mask], labels[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate every epoch
        val_metrics = evaluate_gnn(model, val_graphs, device=device)

        row = {
            'epoch':      epoch,
            'train_loss': avg_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        history.append(row)

        if val_metrics['cross_entropy'] < best_val_loss:
            best_val_loss = val_metrics['cross_entropy']
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)

        if epoch % 5 == 0:
            print(f'  Epoch {epoch:3d} | train_loss={avg_loss:.4f} '
                  f'| val_ce={val_metrics["cross_entropy"]:.4f} '
                  f'| val_top1={val_metrics["top1_acc"]:.3f} '
                  f'| val_top5={val_metrics["top5_acc"]:.3f}')

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_gnn(
    model:  MusicHeteroGNN,
    graphs: List[HeteroData],
    device: str = 'cuda',
    batch_size: int = 32,
    topk:   Tuple[int, ...] = (1, 5, 10),
) -> Dict:
    model.eval()
    hits   = {k: 0 for k in topk}
    total  = 0
    total_loss = 0.0

    for start in range(0, len(graphs), batch_size):
        batch = make_batch(graphs[start:start + batch_size], device)
        logits = model(batch.x_dict, batch.edge_index_dict)
        labels = batch['occ'].y

        mask         = labels != -100
        flat_logits  = logits[mask]
        flat_labels  = labels[mask]

        loss = F.cross_entropy(flat_logits, flat_labels, reduction='sum')
        total_loss += loss.item()
        total      += flat_labels.shape[0]

        for k in topk:
            topk_preds = flat_logits.topk(k, dim=1).indices
            correct    = (topk_preds == flat_labels.unsqueeze(1)).any(dim=1)
            hits[k]   += correct.sum().item()

    if total == 0:
        return {f'top{k}_acc': 0.0 for k in topk}

    results = {f'top{k}_acc': hits[k] / total for k in topk}
    results['cross_entropy'] = total_loss / total
    return results


@torch.no_grad()
def evaluate_gnn_by_section(
    model:  MusicHeteroGNN,
    graphs: List[HeteroData],
    device: str = 'cuda',
) -> Dict[str, Dict]:
    """
    Compute top-1 accuracy broken down by section type.
    Returns dict: section_type → {top1_acc, count}.
    """
    model.eval()

    sec_hits  = {s: 0 for s in SECTION_TYPES}
    sec_total = {s: 0 for s in SECTION_TYPES}

    for g in graphs:
        g = g.to(device)
        logits = model(g.x_dict, g.edge_index_dict)  # [N_occ, C]
        labels = g['occ'].y                           # [N_occ]

        # Derive section type per occ from sec.x
        sec_feats   = g['sec'].x.to(device)              # [N_sec, 13]
        sec_type_ids = sec_feats[:, 2:].argmax(dim=1)    # [N_sec]

        ei_secrev   = g['sec', 'sec_rev', 'occ'].edge_index.to(device)  # [2, N_occ]
        sec_for_occ = torch.zeros(labels.shape[0], dtype=torch.long, device=device)
        sec_for_occ[ei_secrev[1]] = ei_secrev[0]

        occ_sec_type = sec_type_ids[sec_for_occ]   # [N_occ] int → SECTION_TYPES index

        preds = logits.argmax(dim=1)               # [N_occ]
        mask  = labels != -100

        for i in range(labels.shape[0]):
            if not mask[i]:
                continue
            stype_name = SECTION_TYPES[occ_sec_type[i].item()]
            sec_total[stype_name] += 1
            if preds[i].item() == labels[i].item():
                sec_hits[stype_name] += 1

    results = {}
    for s in SECTION_TYPES:
        if sec_total[s] > 0:
            results[s] = {
                'top1_acc': sec_hits[s] / sec_total[s],
                'count':    sec_total[s],
            }
    return results


@torch.no_grad()
def collect_occ_embeddings(
    model:  MusicHeteroGNN,
    graphs: List[HeteroData],
    device: str = 'cuda',
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Extract mean-pooled occ embeddings per song, plus section type labels.
    Returns (embeddings [N_songs, H], song_ids, section_type_ids [N_songs]).
    Used for t-SNE/UMAP visualization.
    """
    model.eval()
    embeddings  = []
    song_ids    = []
    major_secs  = []   # majority section type per song

    for g in graphs:
        g = g.to(device)

        # Hook to get occ embeddings before classifier
        occ_embs = _get_occ_embeddings(model, g)   # [N_occ, H]

        mean_emb = occ_embs.mean(dim=0)
        embeddings.append(mean_emb.cpu())
        song_ids.append(getattr(g, 'song_id', ''))

        # Most common section type in this song
        sec_feats    = g['sec'].x.to(device)
        sec_type_ids = sec_feats[:, 2:].argmax(dim=1)
        ei_secrev    = g['sec', 'sec_rev', 'occ'].edge_index.to(device)
        sec_for_occ  = torch.zeros(occ_embs.shape[0], dtype=torch.long, device=device)
        sec_for_occ[ei_secrev[1]] = ei_secrev[0]
        occ_stypes = sec_type_ids[sec_for_occ]
        # Most frequent section type
        mc = torch.bincount(occ_stypes).argmax().item()
        major_secs.append(mc)

    return (
        torch.stack(embeddings),
        song_ids,
        torch.tensor(major_secs),
    )


def _get_occ_embeddings(model: MusicHeteroGNN, data: HeteroData) -> torch.Tensor:
    """Run forward pass up to (not including) the classifier head."""
    from .graph import CHORD_FEAT_DIM
    device = next(model.parameters()).device
    data = data.to(device)
    x_sec = data['sec'].x
    if not model.use_sec_features:
        x_sec = torch.zeros_like(x_sec)

    occ_input = data['occ'].x
    if model.use_chord_in_occ:
        ei_inst_rev = data.edge_index_dict.get(('chord', 'inst_rev', 'occ'))
        if ei_inst_rev is not None:
            ei_inst_rev = ei_inst_rev.to(device)
            chord_feat_per_occ = torch.zeros(
                occ_input.shape[0], CHORD_FEAT_DIM,
                device=device, dtype=occ_input.dtype,
            )
            chord_feat_per_occ[ei_inst_rev[1]] = data['chord'].x.to(device)[ei_inst_rev[0]]
            occ_input = torch.cat([occ_input, chord_feat_per_occ], dim=1)

    h = {
        'occ':   model.occ_proj(occ_input),
        'chord': model.chord_proj(data['chord'].x),
        'sec':   model.sec_proj(x_sec),
    }
    if model.use_note_edges and model.note_proj is not None:
        h['note'] = model.note_proj(data['note'].x.to(device))
    if model.use_scale_deg_edges and model.scale_deg_proj is not None:
        h['scale_deg'] = model.scale_deg_proj(data['scale_deg'].x.to(device))

    active_ets = set()
    if model.use_seq_edges:
        active_ets.add(('occ', 'next', 'occ'))
    if model.use_seq_edges and model.use_prev_edges:
        active_ets.add(('occ', 'prev', 'occ'))
    if model.use_inst_edges:
        active_ets |= {('occ', 'instance_of', 'chord'), ('chord', 'inst_rev', 'occ')}
    if model.use_section_edges:
        active_ets |= {('occ', 'in_section', 'sec'), ('sec', 'sec_rev', 'occ')}
    if model.use_sec_seq_edges and model.use_section_edges:
        active_ets.add(('sec', 'next_section', 'sec'))
    if model.use_note_edges:
        active_ets |= {('chord', 'chord_contains', 'note'), ('note', 'note_in_chord', 'chord')}
    if model.use_scale_deg_edges:
        active_ets |= {('chord', 'chord_degree', 'scale_deg'), ('scale_deg', 'degree_rev', 'chord')}

    active_edge_index = {
        k: v for k, v in data.edge_index_dict.items()
        if tuple(k) in active_ets or k in active_ets
    }

    for conv, norm in zip(model.convs, model.norms):
        h_new = conv(h, active_edge_index)
        for ntype in h:
            if ntype in h_new:
                h[ntype] = model.dropout(norm(h_new[ntype])) + h[ntype]

    return h['occ']
