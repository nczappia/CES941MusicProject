"""
Non-graph baselines.

MarkovBaseline  — bigram transition model P(next | current, section_type)
LSTMTrainer     — utilities to prepare sequences and train LSTMBaseline
"""

import math
from collections import defaultdict, Counter
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import HeteroData

from .vocab import normalize_chord_to_id, section_type_to_id, VOCAB_SIZE, N_CHORD_ID, NUM_SECTION_TYPES
from .model import LSTMBaseline, NUM_CLASSES


# ---------------------------------------------------------------------------
# Helpers: extract sequences from HeteroData list
# ---------------------------------------------------------------------------

def extract_sequences(graphs: List[HeteroData]):
    """
    Returns two parallel lists of tensors (chord_seq, sec_seq) per song.
    chord_seq[i]: [T_i]  — chord vocab ids for each occurrence
    sec_seq[i]:   [T_i]  — section type ids for each occurrence
    """
    chord_seqs, sec_seqs = [], []
    for g in graphs:
        y = g['occ'].y            # labels: next chord for position i
        # chord ids for positions 0..T-2 are the labels at positions 0..T-2
        # chord id for position 0 is not directly stored; reconstruct from labels
        # Actually: label[i] = chord_id of occurrence i+1
        # So chord_id of occurrence i = label[i-1], and chord_id of occurrence 0
        # requires looking at occ features.  Instead we use labels shifted:
        #   input sequence: positions 0..T-2  → use y[0..T-3] as chord ids? No.
        # Simplest: re-derive chord ids from the graph's instance_of edges.
        # occ i → chord local_j → global chord id via chord.x

        ei = g['occ', 'instance_of', 'chord'].edge_index   # [2, N_occ]
        # ei[0] = occ indices (0..N-1 in order), ei[1] = chord local indices
        chord_local = ei[1]          # [N_occ]
        # chord local → global id: decode from chord.x (one-hot)
        # chord.x[j] = 20-dim one-hot feature; recover chord id from it
        chord_feats = g['chord'].x   # [N_chord_types, 20]
        root_ids    = chord_feats[:, :12].argmax(dim=1)      # [N_chord_types]
        qual_ids    = chord_feats[:, 12:18].argmax(dim=1)
        cplx_ids    = chord_feats[:, 18:20].argmax(dim=1)
        global_ids  = root_ids * 12 + qual_ids * 2 + cplx_ids   # [N_chord_types]

        # Sort by occ index (should already be sorted, but be safe)
        occ_order = ei[0].argsort()
        chord_seq = global_ids[chord_local[occ_order]]       # [N_occ]

        # Section type ids: from graph metadata not directly stored per-occ.
        # Use sec_rev edges: sec_per_occ[i] = section index for occ i
        ei_secrev = g['sec', 'sec_rev', 'occ'].edge_index   # [2, N_occ]
        # ei_secrev[0]=sec idx, ei_secrev[1]=occ idx
        # Build occ→sec map
        sec_for_occ = torch.zeros(chord_seq.shape[0], dtype=torch.long)
        sec_for_occ[ei_secrev[1]] = ei_secrev[0]
        # Decode section type from sec.x one-hot (columns 2..2+11)
        sec_feats = g['sec'].x    # [N_sec, 13]
        sec_type_ids = sec_feats[:, 2:].argmax(dim=1)        # [N_sec]
        sec_seq = sec_type_ids[sec_for_occ]                  # [N_occ]

        chord_seqs.append(chord_seq)
        sec_seqs.append(sec_seq)

    return chord_seqs, sec_seqs


# ---------------------------------------------------------------------------
# Markov baseline
# ---------------------------------------------------------------------------

class MarkovBaseline:
    """
    Bigram + section-conditioned bigram baseline.
    Supports both P(next|curr) and P(next|curr, section).
    """

    def __init__(self, smoothing: float = 1e-6):
        self.smoothing = smoothing
        # (curr_chord_id, sec_type_id) → Counter[next_chord_id]
        self._counts: Dict = defaultdict(Counter)
        # curr_chord_id → Counter[next_chord_id]  (section-marginal)
        self._counts_nosec: Dict = defaultdict(Counter)

    def fit(self, chord_seqs: List[torch.Tensor], sec_seqs: List[torch.Tensor]):
        for cseq, sseq in zip(chord_seqs, sec_seqs):
            clist = cseq.tolist()
            slist = sseq.tolist()
            for i in range(len(clist) - 1):
                curr, nxt, sec = clist[i], clist[i + 1], slist[i]
                self._counts[(curr, sec)][nxt] += 1
                self._counts_nosec[curr][nxt]  += 1

    def predict_topk(
        self,
        curr_chord: int,
        sec_type:   int,
        k: int = 5,
        use_section: bool = True,
    ) -> List[int]:
        if use_section:
            counter = self._counts.get((curr_chord, sec_type),
                      self._counts_nosec.get(curr_chord, Counter()))
        else:
            counter = self._counts_nosec.get(curr_chord, Counter())

        if not counter:
            # uniform fallback
            return list(range(k))

        top = [cid for cid, _ in counter.most_common(k)]
        while len(top) < k:
            top.append(top[-1])
        return top

    def evaluate(
        self,
        chord_seqs: List[torch.Tensor],
        sec_seqs:   List[torch.Tensor],
        topk:       List[int] = (1, 5, 10),
    ) -> Dict:
        hits = {k: 0 for k in topk}
        total = 0
        total_loss = 0.0

        # Build prob distributions
        # vocab size = NUM_CLASSES
        for cseq, sseq in zip(chord_seqs, sec_seqs):
            clist = cseq.tolist()
            slist = sseq.tolist()
            for i in range(len(clist) - 1):
                curr, nxt, sec = clist[i], clist[i + 1], slist[i]
                preds = self.predict_topk(curr, sec, k=max(topk))
                for k in topk:
                    if nxt in preds[:k]:
                        hits[k] += 1
                # Cross-entropy: use smoothed probability
                counter = self._counts.get((curr, sec),
                          self._counts_nosec.get(curr, Counter()))
                total_cnt = sum(counter.values()) + self.smoothing * NUM_CLASSES
                prob = (counter.get(nxt, 0) + self.smoothing) / total_cnt
                total_loss -= math.log(max(prob, 1e-10))
                total += 1

        if total == 0:
            return {f'top{k}_acc': 0.0 for k in topk}

        results = {f'top{k}_acc': hits[k] / total for k in topk}
        results['cross_entropy'] = total_loss / total
        return results


# ---------------------------------------------------------------------------
# LSTM training helpers
# ---------------------------------------------------------------------------

def collate_lstm(
    chord_seqs: List[torch.Tensor],
    sec_seqs:   List[torch.Tensor],
    device:     str = 'cpu',
):
    """
    Pad sequences and return (chord_padded, sec_padded, lengths, label_padded).
    Input at position t  → chord_seqs[t]
    Target at position t → chord_seqs[t+1]
    """
    inputs_c, inputs_s, targets = [], [], []
    for cseq, sseq in zip(chord_seqs, sec_seqs):
        if len(cseq) < 2:
            continue
        inputs_c.append(cseq[:-1])
        inputs_s.append(sseq[:-1])
        targets.append(cseq[1:])

    if not inputs_c:
        return None

    chord_pad  = pad_sequence(inputs_c, batch_first=True, padding_value=0).to(device)
    sec_pad    = pad_sequence(inputs_s, batch_first=True, padding_value=0).to(device)
    target_pad = pad_sequence(targets,  batch_first=True, padding_value=-100).to(device)
    lengths    = torch.tensor([len(s) for s in inputs_c])

    return chord_pad, sec_pad, target_pad, lengths


def train_lstm(
    model:      LSTMBaseline,
    train_seqs: tuple,
    val_seqs:   tuple,
    epochs:     int = 30,
    lr:         float = 1e-3,
    batch_size: int = 32,
    device:     str = 'cpu',
) -> List[Dict]:
    """Train LSTMBaseline, return list of per-epoch metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    model = model.to(device)

    train_c, train_s = train_seqs
    val_c,   val_s   = val_seqs

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        # Shuffle
        perm = torch.randperm(len(train_c))
        train_c = [train_c[i] for i in perm]
        train_s = [train_s[i] for i in perm]

        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, len(train_c), batch_size):
            batch_c = train_c[start:start + batch_size]
            batch_s = train_s[start:start + batch_size]
            batch   = collate_lstm(batch_c, batch_s, device=device)
            if batch is None:
                continue
            chord_in, sec_in, targets, _ = batch

            optimizer.zero_grad()
            logits = model(chord_in, sec_in)   # [B, T, V]
            # Reshape for cross-entropy
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        # Validation
        val_metrics = evaluate_lstm(model, val_c, val_s, device=device)
        avg_loss    = epoch_loss / max(n_batches, 1)
        scheduler.step(val_metrics['cross_entropy'])

        row = {'epoch': epoch, 'train_loss': avg_loss, **val_metrics}
        history.append(row)
        if epoch % 5 == 0:
            print(f'  Epoch {epoch:3d} | train_loss={avg_loss:.4f} '
                  f'| val_ce={val_metrics["cross_entropy"]:.4f} '
                  f'| val_top1={val_metrics["top1_acc"]:.3f} '
                  f'| val_top5={val_metrics["top5_acc"]:.3f}')

    return history


@torch.no_grad()
def evaluate_lstm(
    model:      LSTMBaseline,
    chord_seqs: List[torch.Tensor],
    sec_seqs:   List[torch.Tensor],
    device:     str = 'cpu',
    batch_size: int = 32,
    topk:       tuple = (1, 5, 10),
) -> Dict:
    model.eval()
    hits   = {k: 0 for k in topk}
    total  = 0
    total_loss = 0.0

    for start in range(0, len(chord_seqs), batch_size):
        batch_c = chord_seqs[start:start + batch_size]
        batch_s = sec_seqs[start:start + batch_size]
        batch   = collate_lstm(batch_c, batch_s, device=device)
        if batch is None:
            continue
        chord_in, sec_in, targets, _ = batch

        logits = model(chord_in, sec_in)   # [B, T, V]
        B, T, V = logits.shape

        mask = targets.view(-1) != -100
        flat_logits  = logits.view(B * T, V)[mask]
        flat_targets = targets.view(B * T)[mask]

        loss = F.cross_entropy(flat_logits, flat_targets, reduction='sum')
        total_loss += loss.item()
        total      += flat_targets.shape[0]

        for k in topk:
            topk_preds = flat_logits.topk(k, dim=1).indices  # [N, k]
            correct = (topk_preds == flat_targets.unsqueeze(1)).any(dim=1)
            hits[k] += correct.sum().item()

    if total == 0:
        return {f'top{k}_acc': 0.0 for k in topk}

    results = {f'top{k}_acc': hits[k] / total for k in topk}
    results['cross_entropy'] = total_loss / total
    return results
