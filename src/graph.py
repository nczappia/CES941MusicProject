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
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, List

from .vocab import (
    normalize_chord_to_id,
    chord_id_to_features,
    chord_id_to_pitch_classes,
    chord_str_to_extension_features,
    detect_song_key,
    tonic_to_root,
    transpose_chord_id,
    metre_to_onehot,
    section_type_to_id,
    NUM_SECTION_TYPES,
    METRE_DIM,
    CHORD_EXT_DIM,
    N_CHORD_ID,
    VOCAB_SIZE,
)

# occ features: [duration_norm, time_norm, metre_onehot(5), tonic_onehot(12)] = 19
OCC_FEAT_DIM      = 2 + METRE_DIM + 12    # 19
CHORD_FEAT_DIM    = 20 + CHORD_EXT_DIM   # 25
SEC_FEAT_DIM      = 2 + NUM_SECTION_TYPES # 13
NOTE_FEAT_DIM     = 12
N_NOTES           = 12
SCALE_DEG_FEAT_DIM = 12   # one-hot chromatic scale degree identity (0=tonic..11=maj7)
N_SCALE_DEGREES    = 12


def build_song_heterograph(song: Dict, key_normalize: bool = False) -> HeteroData:
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

    # Optionally transpose all chords to C (key normalization using annotated tonic)
    if key_normalize:
        key_root = tonic_to_root(song.get('tonic', 'C'))
        if key_root != 0:
            chord_ids = [transpose_chord_id(cid, -key_root) for cid in chord_ids]

    unique_cids = sorted(set(chord_ids))
    global_to_local = {gid: lid for lid, gid in enumerate(unique_cids)}
    N_chord = len(unique_cids)

    # chord features: base 20-dim + 5-dim extension flags (parsed from original string)
    unique_cstrs = {cid: chord_strs[chord_ids.index(cid)]
                    for cid in unique_cids}   # map back to one representative string
    chord_feat = torch.stack([
        torch.cat([chord_id_to_features(gid),
                   chord_str_to_extension_features(unique_cstrs[gid])])
        for gid in unique_cids
    ])  # [N_chord, 25]

    # ── occ node features ────────────────────────────────────────────────
    starts    = torch.tensor([c['start'] for c in chords], dtype=torch.float)
    ends      = torch.tensor([c['end']   for c in chords], dtype=torch.float)
    durations = ends - starts

    song_start = starts[0].item()
    song_end   = ends[-1].item()
    song_dur   = max(song_end - song_start, 1e-6)

    dur_norm  = durations / song_dur
    time_norm = (starts - song_start) / song_dur

    # Metre and tonic features — constant across all occs in a song
    metre_feat = metre_to_onehot(song.get('metre', '4/4'))          # [5]
    tonic_root = tonic_to_root(song.get('tonic', 'C'))
    tonic_feat = F.one_hot(torch.tensor(tonic_root), 12).float()    # [12]
    metre_tile = metre_feat.unsqueeze(0).expand(N_occ, -1)          # [N_occ, 5]
    tonic_tile = tonic_feat.unsqueeze(0).expand(N_occ, -1)          # [N_occ, 12]

    occ_feat = torch.cat([
        torch.stack([dur_norm, time_norm], dim=1),   # [N_occ, 2]
        metre_tile,                                   # [N_occ, 5]
        tonic_tile,                                   # [N_occ, 12]
    ], dim=1)   # [N_occ, 19]

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

    # ── note nodes (12 pitch classes, fixed one-hot identity) ────────────
    note_feat = torch.eye(N_NOTES)  # [12, 12]

    # chord → note : chord_contains  (and reverse)
    cn_src, cn_dst = [], []
    for local_id, gid in enumerate(unique_cids):
        for pc in chord_id_to_pitch_classes(gid):
            cn_src.append(local_id)
            cn_dst.append(pc)

    if cn_src:
        ei_chord_contains = torch.tensor([cn_src, cn_dst], dtype=torch.long)
        ei_note_in_chord  = torch.tensor([cn_dst, cn_src], dtype=torch.long)
    else:
        ei_chord_contains = torch.zeros(2, 0, dtype=torch.long)
        ei_note_in_chord  = torch.zeros(2, 0, dtype=torch.long)

    # ── scale degree nodes (12 chromatic degrees relative to tonic) ──────
    # scale_deg[d] = d semitones above the tonic (0=I, 5=IV, 7=V, 10=bVII, etc.)
    scale_deg_feat = torch.eye(N_SCALE_DEGREES)   # [12, 12]

    # chord → scale_deg : chord_degree  (each local chord → its root's scale degree)
    # scale degree = (chord_root - key_root) % 12, using the (possibly transposed) chord_id
    key_root = tonic_to_root(song.get('tonic', 'C'))
    sd_src, sd_dst = [], []
    for local_id, gid in enumerate(unique_cids):
        if gid == N_CHORD_ID:   # no-chord — skip
            continue
        chord_root  = gid // 12
        scale_deg   = (chord_root - key_root) % 12
        sd_src.append(local_id)
        sd_dst.append(scale_deg)

    if sd_src:
        ei_chord_degree = torch.tensor([sd_src, sd_dst], dtype=torch.long)
        ei_degree_rev   = torch.tensor([sd_dst, sd_src], dtype=torch.long)
    else:
        ei_chord_degree = torch.zeros(2, 0, dtype=torch.long)
        ei_degree_rev   = torch.zeros(2, 0, dtype=torch.long)

    # ── assemble HeteroData ──────────────────────────────────────────────
    data = HeteroData()

    data['occ'].x        = occ_feat
    data['occ'].y        = labels
    data['chord'].x      = chord_feat
    data['sec'].x        = sec_feat
    data['note'].x       = note_feat
    data['scale_deg'].x  = scale_deg_feat

    data['occ',       'next',           'occ'].edge_index       = ei_next
    data['occ',       'prev',           'occ'].edge_index       = ei_prev
    data['occ',       'instance_of',    'chord'].edge_index     = ei_inst
    data['chord',     'inst_rev',       'occ'].edge_index       = ei_inst_rev
    data['occ',       'in_section',     'sec'].edge_index       = ei_insec
    data['sec',       'sec_rev',        'occ'].edge_index       = ei_insec_rev
    data['sec',       'next_section',   'sec'].edge_index       = ei_nextsec
    data['chord',     'chord_contains', 'note'].edge_index      = ei_chord_contains
    data['note',      'note_in_chord',  'chord'].edge_index     = ei_note_in_chord
    data['chord',     'chord_degree',   'scale_deg'].edge_index = ei_chord_degree
    data['scale_deg', 'degree_rev',     'chord'].edge_index     = ei_degree_rev

    # Store metadata for downstream use
    data.song_title    = song.get('title', '')
    data.song_artist   = song.get('artist', '')
    data.song_id       = song.get('song_id', '')
    data.num_sections  = N_sec

    return data


# ── Global heterogeneous graph ────────────────────────────────────────────────

SONG_FEAT_DIM = 1   # dummy — song nodes learn from structure


def build_global_heterograph(
    songs: List[Dict],
    split_indices: Dict[str, List[int]],
    key_normalize: bool = True,
) -> 'HeteroData':
    """
    Build one large HeteroData over all songs.

    Node types:
        occ   — all chord occurrences across all songs (concatenated)
        chord — VOCAB_SIZE+1 global chord type nodes (shared across songs)
        sec   — all sections across all songs
        note  — 12 global pitch-class nodes
        song  — one per song

    Edge types (within-song edges are standard; new cross-song edges):
        occ  → song : belongs_to   (each occ → its song node)
        song → occ  : song_rev

    Masks on occ and song nodes:
        train_mask, val_mask, test_mask  (bool tensors)

    split_indices: {'train': [...], 'val': [...], 'test': [...]} — song indices
    """
    N_songs    = len(songs)
    N_chords_g = VOCAB_SIZE + 1   # global chord nodes (0..144)
    N_notes_g  = 12               # global note nodes

    # ── Global chord and note node features ─────────────────────────────────
    # Global chord nodes: base features only (extensions are per-occurrence; for global
    # shared nodes we use zeros for extension flags — they'll be learned via aggregation)
    global_chord_feat = torch.stack([
        torch.cat([chord_id_to_features(i), torch.zeros(CHORD_EXT_DIM)])
        for i in range(N_chords_g)
    ])
    global_note_feat  = torch.eye(N_notes_g)
    song_feat         = torch.zeros(N_songs, SONG_FEAT_DIM)

    # ── Accumulators ─────────────────────────────────────────────────────────
    occ_feats, occ_labels = [], []
    sec_feats             = []

    # Within-song edge accumulators (global indices)
    ei_next_src,     ei_next_dst     = [], []
    ei_prev_src,     ei_prev_dst     = [], []
    ei_inst_src,     ei_inst_dst     = [], []   # occ → global chord
    ei_instrev_src,  ei_instrev_dst  = [], []   # global chord → occ
    ei_insec_src,    ei_insec_dst    = [], []
    ei_insecrev_src, ei_insecrev_dst = [], []
    ei_nextsec_src,  ei_nextsec_dst  = [], []

    # Cross-song edges
    ei_bel_src, ei_bel_dst = [], []   # occ → song
    ei_songrev_src, ei_songrev_dst = [], []

    occ_offset = 0
    sec_offset = 0
    song_ids   = []

    for song_idx, song in enumerate(songs):
        chords   = song['chords']
        sections = song['sections']
        N_occ    = len(chords)
        N_sec    = len(sections)

        chord_strs = [c['chord_str'] for c in chords]
        chord_ids  = [normalize_chord_to_id(cs) for cs in chord_strs]

        if key_normalize:
            key_root = tonic_to_root(song.get('tonic', 'C'))
            if key_root != 0:
                chord_ids = [transpose_chord_id(cid, -key_root) for cid in chord_ids]

        # occ features
        starts    = torch.tensor([c['start'] for c in chords], dtype=torch.float)
        ends      = torch.tensor([c['end']   for c in chords], dtype=torch.float)
        durations = ends - starts
        song_start = starts[0].item(); song_end = ends[-1].item()
        song_dur   = max(song_end - song_start, 1e-6)
        dur_norm   = durations / song_dur
        time_norm  = (starts - song_start) / song_dur

        metre_feat = metre_to_onehot(song.get('metre', '4/4'))
        tonic_root = tonic_to_root(song.get('tonic', 'C'))
        tonic_feat = F.one_hot(torch.tensor(tonic_root), 12).float()
        metre_tile = metre_feat.unsqueeze(0).expand(N_occ, -1)
        tonic_tile = tonic_feat.unsqueeze(0).expand(N_occ, -1)

        occ_feat = torch.cat([
            torch.stack([dur_norm, time_norm], dim=1),
            metre_tile, tonic_tile,
        ], dim=1)   # [N_occ, 19]

        # Inject chord features into occ (use_chord_in_occ always True for global graph)
        chord_feat_per_occ = torch.stack([
            torch.cat([chord_id_to_features(cid),
                       chord_str_to_extension_features(chord_strs[i])])
            for i, cid in enumerate(chord_ids)
        ])   # [N_occ, 25]
        occ_feat = torch.cat([occ_feat, chord_feat_per_occ], dim=1)   # [N_occ, 44]

        occ_feats.append(occ_feat)

        # labels
        labels = torch.full((N_occ,), -100, dtype=torch.long)
        for i in range(N_occ - 1):
            labels[i] = chord_ids[i + 1]
        occ_labels.append(labels)

        # sec features
        for i, sec in enumerate(sections):
            sd = max(sec['end'] - sec['start'], 1e-6) / song_dur
            sp = i / max(N_sec - 1, 1)
            stype_id = section_type_to_id(sec['section_type'])
            onehot = torch.zeros(NUM_SECTION_TYPES)
            onehot[stype_id] = 1.0
            sf = torch.cat([torch.tensor([sd, sp], dtype=torch.float), onehot])
            sec_feats.append(sf)

        # ── Within-song edges (offset into global indices) ───────────────
        idx = torch.arange(N_occ, dtype=torch.long)
        if N_occ > 1:
            ei_next_src.extend((idx[:-1] + occ_offset).tolist())
            ei_next_dst.extend((idx[1:]  + occ_offset).tolist())
            ei_prev_src.extend((idx[1:]  + occ_offset).tolist())
            ei_prev_dst.extend((idx[:-1] + occ_offset).tolist())

        global_cids = torch.tensor(chord_ids, dtype=torch.long)  # already global chord IDs
        ei_inst_src.extend((idx + occ_offset).tolist())
        ei_inst_dst.extend(global_cids.tolist())
        ei_instrev_src.extend(global_cids.tolist())
        ei_instrev_dst.extend((idx + occ_offset).tolist())

        sec_per_occ = torch.tensor(
            [c['section_idx'] for c in chords], dtype=torch.long
        ).clamp(0, N_sec - 1)
        ei_insec_src.extend((idx + occ_offset).tolist())
        ei_insec_dst.extend((sec_per_occ + sec_offset).tolist())
        ei_insecrev_src.extend((sec_per_occ + sec_offset).tolist())
        ei_insecrev_dst.extend((idx + occ_offset).tolist())

        sec_idx = torch.arange(N_sec, dtype=torch.long)
        if N_sec > 1:
            ei_nextsec_src.extend((sec_idx[:-1] + sec_offset).tolist())
            ei_nextsec_dst.extend((sec_idx[1:]  + sec_offset).tolist())

        # ── Cross-song: occ ↔ song ──────────────────────────────────────
        occ_global = (idx + occ_offset).tolist()
        ei_bel_src.extend(occ_global)
        ei_bel_dst.extend([song_idx] * N_occ)
        ei_songrev_src.extend([song_idx] * N_occ)
        ei_songrev_dst.extend(occ_global)

        song_ids.append(song.get('song_id', ''))
        occ_offset += N_occ
        sec_offset += N_sec

    # ── Chord→note edges (global, fixed) ─────────────────────────────────
    cn_src, cn_dst = [], []
    for gid in range(N_chords_g):
        for pc in chord_id_to_pitch_classes(gid):
            cn_src.append(gid); cn_dst.append(pc)

    # ── Split masks ───────────────────────────────────────────────────────
    train_set = set(split_indices['train'])
    val_set   = set(split_indices['val'])
    test_set  = set(split_indices['test'])

    # Map occ → song_idx (we tracked this via ei_bel_dst)
    total_occ = occ_offset
    occ_song  = torch.tensor(ei_bel_dst, dtype=torch.long)  # [total_occ]
    occ_train_mask = torch.tensor([occ_song[i].item() in train_set for i in range(total_occ)])
    occ_val_mask   = torch.tensor([occ_song[i].item() in val_set   for i in range(total_occ)])
    occ_test_mask  = torch.tensor([occ_song[i].item() in test_set  for i in range(total_occ)])

    song_train_mask = torch.tensor([i in train_set for i in range(N_songs)])
    song_val_mask   = torch.tensor([i in val_set   for i in range(N_songs)])
    song_test_mask  = torch.tensor([i in test_set  for i in range(N_songs)])

    # ── Assemble ─────────────────────────────────────────────────────────
    data = HeteroData()

    data['occ'].x          = torch.cat(occ_feats, dim=0)
    data['occ'].y          = torch.cat(occ_labels, dim=0)
    data['occ'].train_mask = occ_train_mask
    data['occ'].val_mask   = occ_val_mask
    data['occ'].test_mask  = occ_test_mask

    data['chord'].x = global_chord_feat
    data['note'].x  = global_note_feat
    data['sec'].x   = torch.stack(sec_feats)
    data['song'].x  = song_feat
    data['song'].train_mask = song_train_mask
    data['song'].val_mask   = song_val_mask
    data['song'].test_mask  = song_test_mask

    def et(src, dst): return torch.tensor([src, dst], dtype=torch.long)

    data['occ',   'next',           'occ'].edge_index   = et(ei_next_src,     ei_next_dst)
    data['occ',   'prev',           'occ'].edge_index   = et(ei_prev_src,     ei_prev_dst)
    data['occ',   'instance_of',    'chord'].edge_index = et(ei_inst_src,     ei_inst_dst)
    data['chord', 'inst_rev',       'occ'].edge_index   = et(ei_instrev_src,  ei_instrev_dst)
    data['occ',   'in_section',     'sec'].edge_index   = et(ei_insec_src,    ei_insec_dst)
    data['sec',   'sec_rev',        'occ'].edge_index   = et(ei_insecrev_src, ei_insecrev_dst)
    data['sec',   'next_section',   'sec'].edge_index   = et(ei_nextsec_src,  ei_nextsec_dst)
    data['chord', 'chord_contains', 'note'].edge_index  = et(cn_src,          cn_dst)
    data['note',  'note_in_chord',  'chord'].edge_index = et(cn_dst,          cn_src)
    data['occ',   'belongs_to',     'song'].edge_index  = et(ei_bel_src,      ei_bel_dst)
    data['song',  'song_rev',       'occ'].edge_index   = et(ei_songrev_src,  ei_songrev_dst)

    data.song_ids = song_ids
    return data
