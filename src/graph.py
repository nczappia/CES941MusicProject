"""
Heterograph construction for a single parsed song.

Graph schema
────────────
Node types
    occ   – one per chord occurrence;  features: [duration_norm, time_norm]
    chord – one per unique chord type appearing in the song; features: 20-dim
    sec   – one per labeled section; features: [dur_norm, pos_norm, sec_type_onehot×11]

Edge types   (all directed)
    occ  → occ   : next           (i → i+1 in time order)
    occ  → occ   : prev           (i+1 → i; lets future context flow back)
    occ  → chord : instance_of    (occurrence → its chord type node)
    chord→ occ   : inst_rev       (reverse; chord type features flow to occ)
    occ  → sec   : in_section     (occurrence → its section node)
    sec  → occ   : sec_rev        (reverse; section features flow to occ)
    sec  → sec   : next_section   (s_j → s_{j+1})

Labels
    data['occ'].y[i] = chord vocab id of occurrence i+1
    data['occ'].y[-1] = -100  (masked out in loss)
"""

import torch
from torch_geometric.data import HeteroData
from typing import Dict, List

from .vocab import (
    normalize_chord_to_id,
    chord_id_to_features,
    section_type_to_id,
    NUM_SECTION_TYPES,
    N_CHORD_ID,
)

OCC_FEAT_DIM   = 2
CHORD_FEAT_DIM = 20
SEC_FEAT_DIM   = 2 + NUM_SECTION_TYPES   # 13


def build_song_heterograph(song: Dict) -> HeteroData:
    """
    song : output of parse.parse_salami_chords()
    Returns a PyG HeteroData object ready for training.
    """
    chords   = song['chords']
    sections = song['sections']
    N_occ    = len(chords)
    N_sec    = len(sections)

    # ── chord type nodes (unique types that appear in this song) ─────────
    chord_strs = [c['chord_str'] for c in chords]
    chord_ids  = [normalize_chord_to_id(cs) for cs in chord_strs]

    unique_cids = sorted(set(chord_ids))
    global_to_local = {gid: lid for lid, gid in enumerate(unique_cids)}
    N_chord = len(unique_cids)

    chord_feat = torch.stack([chord_id_to_features(gid) for gid in unique_cids])  # [N_chord, 20]

    # ── occ node features ────────────────────────────────────────────────
    starts    = torch.tensor([c['start'] for c in chords], dtype=torch.float)
    ends      = torch.tensor([c['end']   for c in chords], dtype=torch.float)
    durations = ends - starts

    song_start = starts[0].item()
    song_end   = ends[-1].item()
    song_dur   = max(song_end - song_start, 1e-6)

    dur_norm  = durations / song_dur
    time_norm = (starts - song_start) / song_dur
    occ_feat  = torch.stack([dur_norm, time_norm], dim=1)   # [N_occ, 2]

    # ── sec node features ────────────────────────────────────────────────
    sec_feats = []
    for i, sec in enumerate(sections):
        sd = max(sec['end'] - sec['start'], 1e-6) / song_dur
        sp = i / max(N_sec - 1, 1)
        stype_id = section_type_to_id(sec['section_type'])
        onehot = torch.zeros(NUM_SECTION_TYPES)
        onehot[stype_id] = 1.0
        sec_feats.append(torch.tensor([sd, sp], dtype=torch.float))
        sec_feats[-1] = torch.cat([sec_feats[-1], onehot])
    sec_feat = torch.stack(sec_feats)   # [N_sec, 13]

    # ── labels: y[i] = chord vocab id of occ i+1 ────────────────────────
    labels = torch.full((N_occ,), -100, dtype=torch.long)
    for i in range(N_occ - 1):
        labels[i] = chord_ids[i + 1]

    # ── edge indices ─────────────────────────────────────────────────────
    idx = torch.arange(N_occ, dtype=torch.long)

    # occ → occ : next
    if N_occ > 1:
        ei_next = torch.stack([idx[:-1], idx[1:]])
        ei_prev = torch.stack([idx[1:],  idx[:-1]])
    else:
        ei_next = torch.zeros(2, 0, dtype=torch.long)
        ei_prev = torch.zeros(2, 0, dtype=torch.long)

    # occ → chord : instance_of  (and reverse)
    local_cids = torch.tensor([global_to_local[cid] for cid in chord_ids], dtype=torch.long)
    ei_inst     = torch.stack([idx, local_cids])
    ei_inst_rev = torch.stack([local_cids, idx])

    # occ → sec : in_section  (and reverse)
    sec_per_occ = torch.tensor([c['section_idx'] for c in chords], dtype=torch.long)
    # Clamp to valid section range (guard against parse edge cases)
    sec_per_occ = sec_per_occ.clamp(0, N_sec - 1)
    ei_insec     = torch.stack([idx, sec_per_occ])
    ei_insec_rev = torch.stack([sec_per_occ, idx])

    # sec → sec : next_section
    sec_idx = torch.arange(N_sec, dtype=torch.long)
    if N_sec > 1:
        ei_nextsec = torch.stack([sec_idx[:-1], sec_idx[1:]])
    else:
        ei_nextsec = torch.zeros(2, 0, dtype=torch.long)

    # ── assemble HeteroData ──────────────────────────────────────────────
    data = HeteroData()

    data['occ'].x  = occ_feat
    data['occ'].y  = labels
    data['chord'].x = chord_feat
    data['sec'].x  = sec_feat

    data['occ',   'next',         'occ'].edge_index   = ei_next
    data['occ',   'prev',         'occ'].edge_index   = ei_prev
    data['occ',   'instance_of',  'chord'].edge_index = ei_inst
    data['chord', 'inst_rev',     'occ'].edge_index   = ei_inst_rev
    data['occ',   'in_section',   'sec'].edge_index   = ei_insec
    data['sec',   'sec_rev',      'occ'].edge_index   = ei_insec_rev
    data['sec',   'next_section', 'sec'].edge_index   = ei_nextsec

    # Store metadata for downstream use
    data.song_title    = song.get('title', '')
    data.song_artist   = song.get('artist', '')
    data.song_id       = song.get('song_id', '')
    data.num_sections  = N_sec

    return data
