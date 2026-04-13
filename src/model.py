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
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv

from .vocab import VOCAB_SIZE, N_CHORD_ID
from .graph import OCC_FEAT_DIM, CHORD_FEAT_DIM, SEC_FEAT_DIM

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
    ):
        super().__init__()
        self.use_seq_edges     = use_seq_edges
        self.use_prev_edges    = use_prev_edges
        self.use_inst_edges    = use_inst_edges
        self.use_section_edges = use_section_edges
        self.use_sec_seq_edges = use_sec_seq_edges
        self.use_sec_features  = use_sec_features
        self.use_chord_in_occ  = use_chord_in_occ
        self.use_attention     = use_attention
        self.gat_heads         = gat_heads

        # Input projections
        # If use_chord_in_occ, occ input = [timing(2) | chord_features(20)] = 22-dim
        occ_in_dim = OCC_FEAT_DIM + (CHORD_FEAT_DIM if use_chord_in_occ else 0)
        self.occ_proj   = nn.Linear(occ_in_dim,      hidden_dim)
        self.chord_proj = nn.Linear(CHORD_FEAT_DIM,  hidden_dim)
        self.sec_proj   = nn.Linear(SEC_FEAT_DIM,    hidden_dim)

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

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, x_dict, edge_index_dict):
        # Zero out sec features if ablated
        x_sec = x_dict['sec']
        if not self.use_sec_features:
            x_sec = torch.zeros_like(x_sec)

        # Optionally inject current chord's features into each occ's input.
        # Uses inst_rev edge (chord→occ) as a lookup — each occ maps to exactly
        # one chord type node, so this is an index gather, not an aggregation.
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

        # Filter edge_index_dict to only edges used by active convs
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

        active_edge_index = {
            k: v for k, v in edge_index_dict.items()
            if tuple(k) in active_ets or k in active_ets
        }

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, active_edge_index)
            # residual + norm + dropout  (only for occ; chord/sec secondary)
            for ntype in h:
                if ntype in h_new:
                    h[ntype] = self.dropout(norm(h_new[ntype])) + h[ntype]

        return self.classifier(h['occ'])   # [N_occ_in_batch, NUM_CLASSES]


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
