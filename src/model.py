"""
Heterogeneous GNN for next-chord prediction.

Architecture
────────────
1. Linear projections: occ(2-d), chord(20-d), sec(13-d) → hidden_dim each
2. N layers of HeteroConv (SAGEConv or GATConv per relation)
3. Linear classifier: occ embedding → 145 logits (144 chord types + N)

Ablation flags
──────────────
    use_seq_edges     : (occ,next/prev,occ)
    use_inst_edges    : (occ,instance_of,chord) + (chord,inst_rev,occ)
    use_section_edges : (occ,in_section,sec) + (sec,sec_rev,occ)
    use_sec_seq_edges : (sec,next_section,sec)
    use_sec_features  : if False, zero out sec node features
    use_attention     : if True, use GATConv instead of SAGEConv (enables attention extraction)

Also contains LSTMBaseline for sequence-model comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, HGTConv

from .vocab import VOCAB_SIZE, N_CHORD_ID
from .graph import OCC_FEAT_DIM, CHORD_FEAT_DIM, SEC_FEAT_DIM, NOTE_FEAT_DIM, SCALE_DEG_FEAT_DIM

NUM_CLASSES = VOCAB_SIZE + 1   # 144 chord types + 1 no-chord = 145


class MusicHeteroGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        # ablation switches
        use_seq_edges: bool     = True,
        use_prev_edges: bool    = True,   # False = causal model (no future leakage)
        use_inst_edges: bool    = True,
        use_section_edges: bool = True,
        use_sec_seq_edges: bool = True,
        use_sec_features: bool  = True,
        # input enrichment
        use_chord_in_occ: bool  = True,   # inject current chord features into occ input
        # attention
        use_attention: bool     = False,  # True = GATConv instead of SAGEConv
        gat_heads: int          = 4,      # number of attention heads
        # note-level graph extension
        use_note_edges: bool      = False,  # True = add note nodes + chord_contains/note_in_chord edges
        # scale degree graph extension
        use_scale_deg_edges: bool = False,  # True = add scale_deg nodes + chord_degree/degree_rev edges
        # multi-task
        num_genres: int           = 0,      # >0 = add song-level genre classification head
    ):
        super().__init__()
        self.use_seq_edges     = use_seq_edges
        self.use_prev_edges    = use_prev_edges
        self.use_inst_edges    = use_inst_edges
        self.use_section_edges = use_section_edges
        self.use_sec_seq_edges = use_sec_seq_edges
        self.use_sec_features  = use_sec_features
        self.use_chord_in_occ  = use_chord_in_occ
        self.use_attention        = use_attention
        self.gat_heads            = gat_heads
        self.use_note_edges       = use_note_edges
        self.use_scale_deg_edges  = use_scale_deg_edges

        # Input projections
        # If use_chord_in_occ, occ input = [timing(2) | chord_features(20)] = 22-dim
        occ_in_dim = OCC_FEAT_DIM + (CHORD_FEAT_DIM if use_chord_in_occ else 0)
        self.occ_proj        = nn.Linear(occ_in_dim,          hidden_dim)
        self.chord_proj      = nn.Linear(CHORD_FEAT_DIM,      hidden_dim)
        self.sec_proj        = nn.Linear(SEC_FEAT_DIM,        hidden_dim)
        self.note_proj       = nn.Linear(NOTE_FEAT_DIM,       hidden_dim) if use_note_edges else None
        self.scale_deg_proj  = nn.Linear(SCALE_DEG_FEAT_DIM,  hidden_dim) if use_scale_deg_edges else None

        # HeteroConv layers
        # GATConv: multi-head attention; output dim = hidden_dim // heads per head, concat → hidden_dim
        assert hidden_dim % gat_heads == 0, "hidden_dim must be divisible by gat_heads"
        gat_out = hidden_dim // gat_heads  # per-head dim; concat gives hidden_dim

        def make_conv(in_dim=hidden_dim, out_dim=hidden_dim):
            if use_attention:
                return GATConv(in_dim, gat_out, heads=gat_heads,
                               concat=True, dropout=dropout, add_self_loops=False)
            return SAGEConv(in_dim, out_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            if use_seq_edges:
                conv_dict[('occ', 'next', 'occ')] = make_conv()
            if use_seq_edges and use_prev_edges:
                conv_dict[('occ', 'prev', 'occ')] = make_conv()
            if use_inst_edges:
                conv_dict[('occ',   'instance_of', 'chord')] = make_conv()
                conv_dict[('chord', 'inst_rev',    'occ')]   = make_conv()
            if use_section_edges:
                conv_dict[('occ', 'in_section', 'sec')] = make_conv()
                conv_dict[('sec', 'sec_rev',    'occ')] = make_conv()
            if use_sec_seq_edges and use_section_edges:
                conv_dict[('sec', 'next_section', 'sec')] = make_conv()
            if use_note_edges:
                conv_dict[('chord', 'chord_contains', 'note')] = make_conv()
                conv_dict[('note',  'note_in_chord',  'chord')] = make_conv()
            if use_scale_deg_edges:
                conv_dict[('chord',     'chord_degree', 'scale_deg')] = make_conv()
                conv_dict[('scale_deg', 'degree_rev',   'chord')]     = make_conv()

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)
        self.genre_head = nn.Linear(hidden_dim, num_genres) if num_genres > 0 else None
        self.hidden_dim = hidden_dim

    def _encode(self, x_dict, edge_index_dict):
        """Shared forward logic. Returns h dict with final node embeddings."""
        x_sec = x_dict['sec']
        if not self.use_sec_features:
            x_sec = torch.zeros_like(x_sec)

        occ_input = x_dict['occ']
        if self.use_chord_in_occ:
            ei_inst_rev = edge_index_dict.get(('chord', 'inst_rev', 'occ'))
            if ei_inst_rev is not None:
                chord_feat_per_occ = torch.zeros(
                    occ_input.shape[0], CHORD_FEAT_DIM,
                    device=occ_input.device, dtype=occ_input.dtype,
                )
                chord_feat_per_occ[ei_inst_rev[1]] = x_dict['chord'][ei_inst_rev[0]]
                occ_input = torch.cat([occ_input, chord_feat_per_occ], dim=1)

        h = {
            'occ':   self.occ_proj(occ_input),
            'chord': self.chord_proj(x_dict['chord']),
            'sec':   self.sec_proj(x_sec),
        }
        if self.use_note_edges and self.note_proj is not None:
            h['note'] = self.note_proj(x_dict['note'])
        if self.use_scale_deg_edges and self.scale_deg_proj is not None:
            h['scale_deg'] = self.scale_deg_proj(x_dict['scale_deg'])

        active_ets = set()
        if self.use_seq_edges:
            active_ets.add(('occ', 'next', 'occ'))
        if self.use_seq_edges and self.use_prev_edges:
            active_ets.add(('occ', 'prev', 'occ'))
        if self.use_inst_edges:
            active_ets |= {('occ', 'instance_of', 'chord'), ('chord', 'inst_rev', 'occ')}
        if self.use_section_edges:
            active_ets |= {('occ', 'in_section', 'sec'), ('sec', 'sec_rev', 'occ')}
        if self.use_sec_seq_edges and self.use_section_edges:
            active_ets.add(('sec', 'next_section', 'sec'))
        if self.use_note_edges:
            active_ets |= {('chord', 'chord_contains', 'note'), ('note', 'note_in_chord', 'chord')}
        if self.use_scale_deg_edges:
            active_ets |= {('chord', 'chord_degree', 'scale_deg'), ('scale_deg', 'degree_rev', 'chord')}

        active_edge_index = {
            k: v for k, v in edge_index_dict.items()
            if tuple(k) in active_ets or k in active_ets
        }

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, active_edge_index)
            for ntype in h:
                if ntype in h_new:
                    h[ntype] = self.dropout(norm(h_new[ntype])) + h[ntype]

        return h

    def forward(self, x_dict, edge_index_dict):
        return self.classifier(self._encode(x_dict, edge_index_dict)['occ'])

    def encode_occ(self, x_dict, edge_index_dict):
        """Return occ node embeddings [N_occ, hidden_dim] before the classifier."""
        return self._encode(x_dict, edge_index_dict)['occ']

    def forward_with_genre(self, x_dict, edge_index_dict, occ_batch=None):
        """
        Like forward(), but also returns song-level genre logits.
        occ_batch : [N_occ] LongTensor mapping each occ to its song index in the batch.
                    When None, treats all occs as one song.
        Returns: (chord_logits [N_occ, C], genre_logits [B, num_genres])
        """
        from torch_scatter import scatter_mean
        assert self.genre_head is not None, "num_genres=0 — no genre head"

        h = self._encode(x_dict, edge_index_dict)
        chord_logits = self.classifier(h['occ'])

        if occ_batch is None:
            occ_batch = torch.zeros(h['occ'].shape[0], dtype=torch.long, device=h['occ'].device)
        song_emb     = scatter_mean(h['occ'], occ_batch, dim=0)
        genre_logits = self.genre_head(song_emb)

        return chord_logits, genre_logits


# ---------------------------------------------------------------------------
# Heterogeneous Graph Transformer (HGT, Hu et al. 2020)
# ---------------------------------------------------------------------------

class MusicHGT(nn.Module):
    """
    Heterogeneous Graph Transformer for next-chord prediction.

    HGTConv applies type-specific linear transforms and scaled dot-product
    attention across every (src_type, rel, dst_type) triple simultaneously.
    This is strictly more expressive than SAGEConv for heterogeneous graphs
    because attention weights are learned per node-type pair.

    Always causal (no prev edges).
    Always uses all five node types: occ, chord, sec, note, scale_deg.
    Genre head uses hierarchical sec-level pooling: after the GNN,
    h['sec'] has aggregated occ information; we mean-pool sec per song.
    """

    _NODE_TYPES = ['occ', 'chord', 'sec', 'note', 'scale_deg']
    _EDGE_TYPES = [
        ('occ',       'next',           'occ'),
        ('occ',       'instance_of',    'chord'),
        ('chord',     'inst_rev',       'occ'),
        ('occ',       'in_section',     'sec'),
        ('sec',       'sec_rev',        'occ'),
        ('sec',       'next_section',   'sec'),
        ('chord',     'chord_contains', 'note'),
        ('note',      'note_in_chord',  'chord'),
        ('chord',     'chord_degree',   'scale_deg'),
        ('scale_deg', 'degree_rev',     'chord'),
    ]

    def __init__(
        self,
        hidden_dim: int  = 128,
        num_layers: int  = 3,
        dropout:    float = 0.3,
        hgt_heads:  int  = 4,
        num_genres: int  = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_genres = num_genres

        # Input projections (always chord-in-occ)
        occ_in_dim = OCC_FEAT_DIM + CHORD_FEAT_DIM   # 19 + 25 = 44
        self.occ_proj       = nn.Linear(occ_in_dim,         hidden_dim)
        self.chord_proj     = nn.Linear(CHORD_FEAT_DIM,     hidden_dim)
        self.sec_proj       = nn.Linear(SEC_FEAT_DIM,       hidden_dim)
        self.note_proj      = nn.Linear(NOTE_FEAT_DIM,      hidden_dim)
        self.scale_deg_proj = nn.Linear(SCALE_DEG_FEAT_DIM, hidden_dim)

        metadata = (self._NODE_TYPES, self._EDGE_TYPES)
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=hgt_heads)
            for _ in range(num_layers)
        ])
        self.norms      = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)
        self.genre_head = nn.Linear(hidden_dim, num_genres) if num_genres > 0 else None

    def _build_h(self, x_dict, edge_index_dict):
        """Project all node types to hidden_dim, injecting chord feats into occ."""
        ei_inst_rev = edge_index_dict.get(('chord', 'inst_rev', 'occ'))
        occ_input = x_dict['occ']
        if ei_inst_rev is not None:
            chord_feat_per_occ = torch.zeros(
                occ_input.shape[0], CHORD_FEAT_DIM,
                device=occ_input.device, dtype=occ_input.dtype,
            )
            chord_feat_per_occ[ei_inst_rev[1]] = x_dict['chord'][ei_inst_rev[0]]
            occ_input = torch.cat([occ_input, chord_feat_per_occ], dim=1)

        return {
            'occ':       self.occ_proj(occ_input),
            'chord':     self.chord_proj(x_dict['chord']),
            'sec':       self.sec_proj(x_dict['sec']),
            'note':      self.note_proj(x_dict['note']),
            'scale_deg': self.scale_deg_proj(x_dict['scale_deg']),
        }

    def _filter_edges(self, edge_index_dict):
        """Keep only edge types defined in HGT metadata."""
        active = set(map(tuple, self._EDGE_TYPES))
        return {k: v for k, v in edge_index_dict.items()
                if tuple(k) in active or k in active}

    def _encode(self, x_dict, edge_index_dict):
        h  = self._build_h(x_dict, edge_index_dict)
        ei = self._filter_edges(edge_index_dict)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, ei)
            for ntype in h:
                if ntype in h_new and h_new[ntype] is not None:
                    h[ntype] = self.dropout(norm(h_new[ntype])) + h[ntype]
        return h

    def forward(self, x_dict, edge_index_dict):
        h = self._encode(x_dict, edge_index_dict)
        return self.classifier(h['occ'])

    def forward_with_genre(self, x_dict, edge_index_dict,
                           occ_batch=None, sec_batch=None):
        """
        Returns (chord_logits [N_occ, C], genre_logits [B, num_genres]).

        Genre uses hierarchical pooling: after the GNN, h['sec'] has absorbed
        occ-level information via in_section/sec_rev message passing.
        We mean-pool sec embeddings per song — richer than flat occ pooling
        because sections (verse, chorus, bridge) carry distinct harmonic roles.

        occ_batch : [N_occ] LongTensor — song index per occ (from Batch)
        sec_batch : [N_sec] LongTensor — song index per sec (from Batch)
        """
        from torch_scatter import scatter_mean
        assert self.genre_head is not None, "num_genres=0 — no genre head"

        h = self._encode(x_dict, edge_index_dict)
        chord_logits = self.classifier(h['occ'])

        # Hierarchical sec → song pooling
        n_songs = int(occ_batch.max().item() + 1) if occ_batch is not None else 1
        if sec_batch is None:
            sec_batch = torch.zeros(h['sec'].shape[0], dtype=torch.long,
                                    device=h['sec'].device)
        song_emb     = scatter_mean(h['sec'], sec_batch, dim=0, dim_size=n_songs)
        genre_logits = self.genre_head(song_emb)

        return chord_logits, genre_logits


# ---------------------------------------------------------------------------
# Transformer baseline (causal sequence model)
# ---------------------------------------------------------------------------

class TransformerBaseline(nn.Module):
    """
    Causal Transformer decoder for next-chord prediction on chord sequences.

    Drop-in replacement for LSTMBaseline: same forward(chord_ids, sec_ids) → [B, T, V]
    interface, so collate_lstm / evaluate_lstm from baselines.py work unchanged.

    Configuration used in script 16:
        embed_dim=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1
    """

    def __init__(
        self,
        vocab_size:      int   = NUM_CLASSES,
        embed_dim:       int   = 128,
        nhead:           int   = 4,
        num_layers:      int   = 3,
        dim_feedforward: int   = 512,
        dropout:         float = 0.1,
        num_sections:    int   = 11,
        max_len:         int   = 2000,
    ):
        super().__init__()
        self.chord_embed   = nn.Embedding(vocab_size, embed_dim)
        self.section_embed = nn.Embedding(num_sections, embed_dim)
        self.pos_embed     = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier  = nn.Linear(embed_dim, vocab_size)

    def _hidden(self, chord_ids, sec_ids):
        T   = chord_ids.shape[1]
        pos = torch.arange(T, device=chord_ids.device).unsqueeze(0)
        h   = (self.chord_embed(chord_ids)
               + self.section_embed(sec_ids)
               + self.pos_embed(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=chord_ids.device)
        return self.transformer(h, mask=mask, is_causal=True)   # [B, T, embed_dim]

    def forward(self, chord_ids, sec_ids):
        return self.classifier(self._hidden(chord_ids, sec_ids))   # [B, T, V]

    def encode_song(self, chord_ids, sec_ids):
        """Mean-pool over time → [B, embed_dim] song-level embedding."""
        return self._hidden(chord_ids, sec_ids).mean(dim=1)


# ---------------------------------------------------------------------------
# Homogeneous GNN baseline
# ---------------------------------------------------------------------------

class HomoMusicGNN(nn.Module):
    """
    Homogeneous GNN baseline for ablation against MusicHeteroGNN.

    Uses the same causal edge set and chord-in-occ input enrichment as
    causal v2, but collapses all node/edge types into a single type for
    message passing.  This isolates the contribution of heterogeneous
    type-specialization in the conv layers.

    Design:
    - Type-specific input projections are kept (different feature dims require it).
    - A learnable node-type embedding is added post-projection so the model
      can still distinguish node roles.
    - A single SAGEConv per layer aggregates all edges identically.
    """

    _NODE_TYPES = ['occ', 'chord', 'sec', 'note', 'scale_deg']
    _EDGE_TYPES = [
        ('occ',       'next',           'occ'),
        ('occ',       'instance_of',    'chord'),
        ('chord',     'inst_rev',       'occ'),
        ('occ',       'in_section',     'sec'),
        ('sec',       'sec_rev',        'occ'),
        ('sec',       'next_section',   'sec'),
        ('chord',     'chord_contains', 'note'),
        ('note',      'note_in_chord',  'chord'),
        ('chord',     'chord_degree',   'scale_deg'),
        ('scale_deg', 'degree_rev',     'chord'),
    ]

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        occ_in_dim = OCC_FEAT_DIM + CHORD_FEAT_DIM   # always chord-in-occ (causal v2 config)
        self.occ_proj       = nn.Linear(occ_in_dim,         hidden_dim)
        self.chord_proj     = nn.Linear(CHORD_FEAT_DIM,     hidden_dim)
        self.sec_proj       = nn.Linear(SEC_FEAT_DIM,       hidden_dim)
        self.note_proj      = nn.Linear(NOTE_FEAT_DIM,      hidden_dim)
        self.scale_deg_proj = nn.Linear(SCALE_DEG_FEAT_DIM, hidden_dim)

        self.type_embed = nn.Embedding(len(self._NODE_TYPES), hidden_dim)

        self.convs      = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms      = nn.ModuleList([nn.LayerNorm(hidden_dim)         for _ in range(num_layers)])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)

    def _forward_full(self, x_dict, edge_index_dict):
        device = next(self.parameters()).device

        # Chord-in-occ injection (same as causal v2)
        occ_input   = x_dict['occ']
        ei_inst_rev = edge_index_dict.get(('chord', 'inst_rev', 'occ'))
        if ei_inst_rev is not None:
            chord_feat_per_occ = torch.zeros(
                occ_input.shape[0], CHORD_FEAT_DIM,
                device=occ_input.device, dtype=occ_input.dtype,
            )
            chord_feat_per_occ[ei_inst_rev[1]] = x_dict['chord'][ei_inst_rev[0]]
            occ_input = torch.cat([occ_input, chord_feat_per_occ], dim=1)

        type2id = {t: i for i, t in enumerate(self._NODE_TYPES)}

        def te(name):
            return self.type_embed(torch.tensor(type2id[name], device=device))

        h_parts = [
            self.occ_proj(occ_input)                + te('occ'),
            self.chord_proj(x_dict['chord'])        + te('chord'),
            self.sec_proj(x_dict['sec'])            + te('sec'),
            self.note_proj(x_dict['note'])          + te('note'),
            self.scale_deg_proj(x_dict['scale_deg']) + te('scale_deg'),
        ]

        n = [p.shape[0] for p in h_parts]
        offsets = {'occ': 0, 'chord': n[0], 'sec': n[0]+n[1],
                   'note': n[0]+n[1]+n[2], 'scale_deg': n[0]+n[1]+n[2]+n[3]}

        h = torch.cat(h_parts, dim=0)   # [N_total, hidden_dim]

        active = set(map(tuple, self._EDGE_TYPES))
        edge_parts = []
        for et, ei in edge_index_dict.items():
            if tuple(et) not in active:
                continue
            src_type, _, dst_type = et
            shifted = ei.clone()
            shifted[0] = shifted[0] + offsets[src_type]
            shifted[1] = shifted[1] + offsets[dst_type]
            edge_parts.append(shifted)

        edge_index = (torch.cat(edge_parts, dim=1) if edge_parts
                      else torch.zeros(2, 0, dtype=torch.long, device=device))

        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(norm(conv(h, edge_index))) + h

        return self.classifier(h[:n[0]]), h[:n[0]], n[0]   # logits, occ_emb, n_occ

    def forward(self, x_dict, edge_index_dict):
        logits, _, _ = self._forward_full(x_dict, edge_index_dict)
        return logits

    def encode_occ(self, x_dict, edge_index_dict):
        """Returns occ node embeddings [N_occ, hidden_dim] before the classifier."""
        _, occ_emb, _ = self._forward_full(x_dict, edge_index_dict)
        return occ_emb


# ---------------------------------------------------------------------------
# Global heterogeneous GNN (cross-song shared chord nodes + song nodes)
# ---------------------------------------------------------------------------

class GlobalMusicGNN(nn.Module):
    """
    GNN trained on a single global heterogeneous graph across all songs.

    Differences from MusicHeteroGNN:
    - Chord nodes are global (VOCAB_SIZE+1 shared across all songs)
    - Song nodes aggregate occ embeddings via belongs_to/song_rev edges
    - Genre head operates directly on song node embeddings (no scatter_mean needed)
    - occ input always includes chord features (use_chord_in_occ always True)
    - Always causal (no prev edges)
    - Always uses note edges
    """
    def __init__(
        self,
        hidden_dim:  int = 128,
        num_layers:  int = 3,
        dropout:     float = 0.3,
        num_genres:  int = 0,
    ):
        super().__init__()
        from .graph import OCC_FEAT_DIM, CHORD_FEAT_DIM, SEC_FEAT_DIM, NOTE_FEAT_DIM, SONG_FEAT_DIM
        self.hidden_dim = hidden_dim
        self.num_genres = num_genres

        occ_in_dim = OCC_FEAT_DIM + CHORD_FEAT_DIM   # 2 + 20 = 22

        self.occ_proj   = nn.Linear(occ_in_dim,     hidden_dim)
        self.chord_proj = nn.Linear(CHORD_FEAT_DIM,  hidden_dim)
        self.sec_proj   = nn.Linear(SEC_FEAT_DIM,    hidden_dim)
        self.note_proj  = nn.Linear(NOTE_FEAT_DIM,   hidden_dim)
        self.song_proj  = nn.Linear(SONG_FEAT_DIM,   hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ('occ',   'next',           'occ'):   SAGEConv(hidden_dim, hidden_dim),
                ('occ',   'instance_of',    'chord'): SAGEConv(hidden_dim, hidden_dim),
                ('chord', 'inst_rev',       'occ'):   SAGEConv(hidden_dim, hidden_dim),
                ('occ',   'in_section',     'sec'):   SAGEConv(hidden_dim, hidden_dim),
                ('sec',   'sec_rev',        'occ'):   SAGEConv(hidden_dim, hidden_dim),
                ('sec',   'next_section',   'sec'):   SAGEConv(hidden_dim, hidden_dim),
                ('chord', 'chord_contains', 'note'):  SAGEConv(hidden_dim, hidden_dim),
                ('note',  'note_in_chord',  'chord'): SAGEConv(hidden_dim, hidden_dim),
                ('occ',   'belongs_to',     'song'):  SAGEConv(hidden_dim, hidden_dim),
                ('song',  'song_rev',       'occ'):   SAGEConv(hidden_dim, hidden_dim),
            }
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        self.norms      = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)
        self.genre_head = nn.Linear(hidden_dim, num_genres) if num_genres > 0 else None

    def _encode(self, x_dict, edge_index_dict):
        h = {
            'occ':   self.occ_proj(x_dict['occ']),
            'chord': self.chord_proj(x_dict['chord']),
            'sec':   self.sec_proj(x_dict['sec']),
            'note':  self.note_proj(x_dict['note']),
            'song':  self.song_proj(x_dict['song']),
        }
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index_dict)
            for ntype in h:
                if ntype in h_new:
                    h[ntype] = self.dropout(norm(h_new[ntype])) + h[ntype]
        return h

    def forward(self, x_dict, edge_index_dict):
        h = self._encode(x_dict, edge_index_dict)
        return self.classifier(h['occ'])

    def forward_with_genre(self, x_dict, edge_index_dict):
        """Returns (chord_logits [N_occ, C], genre_logits [N_song, num_genres])."""
        assert self.genre_head is not None
        h = self._encode(x_dict, edge_index_dict)
        return self.classifier(h['occ']), self.genre_head(h['song'])


# ---------------------------------------------------------------------------
# LSTM baseline
# ---------------------------------------------------------------------------

class LSTMBaseline(nn.Module):
    """
    Sequence model baseline: embed (chord_id, section_type_id) → LSTM → classify.
    Trained with teacher forcing: at step i, predict chord at i+1.
    """
    def __init__(
        self,
        vocab_size:     int = NUM_CLASSES,
        sec_vocab_size: int = 11,
        embed_dim:      int = 64,
        hidden_dim:     int = 128,
        num_layers:     int = 2,
        dropout:        float = 0.3,
    ):
        super().__init__()
        self.chord_embed = nn.Embedding(vocab_size,     embed_dim, padding_idx=0)
        self.sec_embed   = nn.Embedding(sec_vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim * 2, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, chord_ids: torch.Tensor, sec_ids: torch.Tensor):
        """
        chord_ids : [B, T]  (padded sequences)
        sec_ids   : [B, T]
        Returns logits [B, T, vocab_size].
        """
        ce = self.chord_embed(chord_ids)     # [B, T, E]
        se = self.sec_embed(sec_ids)         # [B, T, E]
        x  = self.dropout(torch.cat([ce, se], dim=-1))   # [B, T, 2E]
        out, _ = self.lstm(x)                # [B, T, H]
        return self.classifier(out)          # [B, T, V]
