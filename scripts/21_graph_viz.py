"""
Script 21 — Graph structure diagrams.

Produces:
    results/graph_schema.png   — annotated schema: node types, features, edge relations
    results/graph_instance.png — actual graph for one real song (small excerpt)

Run from project root:
    source venv/bin/activate && python scripts/21_graph_viz.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import networkx as nx
from pathlib import Path

RESULTS = 'results'
Path(RESULTS).mkdir(exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_OCC       = '#4C9BE8'   # blue
C_CHORD     = '#4CAF50'   # green
C_SEC       = '#FF9800'   # orange
C_NOTE      = '#AB47BC'   # purple
C_SCALEDEG  = '#EF5350'   # red
C_BG        = '#0F0F1A'
C_TEXT      = '#EEEEEE'

NODE_COLORS = {
    'occ': C_OCC, 'chord': C_CHORD, 'sec': C_SEC,
    'note': C_NOTE, 'scale_deg': C_SCALEDEG,
}

EDGE_COLORS = {
    'next':           '#4C9BE8',
    'instance_of':    '#4CAF50',
    'inst_rev':       '#81C784',
    'in_section':     '#FF9800',
    'sec_rev':        '#FFB74D',
    'next_section':   '#FF7043',
    'chord_contains': '#AB47BC',
    'note_in_chord':  '#CE93D8',
    'chord_degree':   '#EF5350',
    'degree_rev':     '#EF9A9A',
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Schema diagram
# ══════════════════════════════════════════════════════════════════════════════

def draw_schema():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis('off')

    # ── Node boxes ────────────────────────────────────────────────────────────
    nodes = {
        'occ': {
            'xy': (4.0, 5.5), 'w': 3.6, 'h': 3.0,
            'color': C_OCC,
            'title': 'occ',
            'subtitle': 'one per chord occurrence\n~120 per song',
            'feats': [
                'duration_norm     — fraction of song duration',
                'time_norm         — onset position in song (0→1)',
                'metre [5-d]       — one-hot: 4/4, 12/8, 3/4, 6/8, other',
                'tonic [12-d]      — one-hot key root (C=0 … B=11)',
                '+ chord feats     — injected at input (causal model only)',
                '                    [25-d, same as chord node]',
            ],
            'dim': '19-d  (44-d w/ chord injection)',
        },
        'chord': {
            'xy': (4.0, 1.5), 'w': 3.6, 'h': 2.6,
            'color': C_CHORD,
            'title': 'chord',
            'subtitle': 'one per unique chord type\nin the song  (~8–30)',
            'feats': [
                'root [12-d]       — one-hot pitch class (C … B)',
                'quality [6-d]     — maj / min / dim / aug / sus / other',
                'complexity [2-d]  — triad vs. seventh-or-more',
                'is_dom7           — binary extension flag',
                'is_maj7           — binary extension flag',
                'is_min7           — binary extension flag',
                'is_sus            — binary extension flag',
                'is_extended       — binary (9th, 11th, 13th present)',
            ],
            'dim': '25-d  (12 + 6 + 2 + 5)',
        },
        'sec': {
            'xy': (11.0, 7.0), 'w': 3.6, 'h': 2.0,
            'color': C_SEC,
            'title': 'sec',
            'subtitle': 'one per labeled section\n(verse, chorus, bridge …)',
            'feats': [
                'dur_norm          — section duration / song duration',
                'pos_norm          — index / (N_sec − 1)',
                'type [11-d]       — one-hot: intro, verse, chorus,',
                '                    bridge, prechorus, outro,',
                '                    instrumental, interlude,',
                '                    solo, trans, other',
            ],
            'dim': '13-d  (2 + 11)',
        },
        'note': {
            'xy': (11.0, 4.0), 'w': 3.0, 'h': 1.4,
            'color': C_NOTE,
            'title': 'note',
            'subtitle': '12 global pitch-class nodes\n(C, C#, D … B)',
            'feats': [
                'identity [12-d]   — fixed one-hot (note index)',
            ],
            'dim': '12-d  (shared across all songs)',
        },
        'scale_deg': {
            'xy': (11.0, 2.0), 'w': 3.0, 'h': 1.4,
            'color': C_SCALEDEG,
            'title': 'scale_deg',
            'subtitle': '12 global scale-degree nodes\n(I, ♭II, II … VII)',
            'feats': [
                'identity [12-d]   — fixed one-hot (semitones above tonic)',
            ],
            'dim': '12-d  (shared across all songs)',
        },
    }

    box_handles = {}
    for ntype, nd in nodes.items():
        x, y = nd['xy']
        w, h = nd['w'], nd['h']
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle='round,pad=0.05',
                              facecolor=nd['color'] + '33',
                              edgecolor=nd['color'], linewidth=2.5,
                              zorder=2)
        ax.add_patch(rect)
        box_handles[ntype] = rect

        # Title
        ax.text(x, y + h/2 - 0.22, nd['title'],
                ha='center', va='top', fontsize=14, fontweight='bold',
                color=nd['color'], zorder=3)
        ax.text(x, y + h/2 - 0.48, nd['subtitle'],
                ha='center', va='top', fontsize=8.5, color=C_TEXT,
                style='italic', zorder=3)

        # Feature list
        feat_y = y + h/2 - 0.85
        for feat in nd['feats']:
            ax.text(x - w/2 + 0.12, feat_y, feat,
                    ha='left', va='top', fontsize=7.2, color=C_TEXT,
                    fontfamily='monospace', zorder=3)
            feat_y -= 0.27

        # Dim badge
        ax.text(x, y - h/2 + 0.12, nd['dim'],
                ha='center', va='bottom', fontsize=8, color=nd['color'],
                fontweight='bold', zorder=3)

    # ── Edges (arrows between node boxes) ────────────────────────────────────
    def node_edge_pt(ntype, direction):
        """Return the anchor point on the edge of a node box."""
        nd = nodes[ntype]
        x, y = nd['xy']; w, h = nd['w'], nd['h']
        if direction == 'right':  return (x + w/2, y)
        if direction == 'left':   return (x - w/2, y)
        if direction == 'top':    return (x, y + h/2)
        if direction == 'bottom': return (x, y - h/2)

    arrow_style = dict(arrowstyle='->', color='white', lw=1.5,
                       connectionstyle='arc3,rad=0.0',
                       zorder=4,
                       mutation_scale=14)

    def draw_arrow(src_pt, dst_pt, label, color, rad=0.0, lbl_offset=(0, 0)):
        ax.annotate('', xy=dst_pt, xytext=src_pt,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                   connectionstyle=f'arc3,rad={rad}'))
        mx = (src_pt[0] + dst_pt[0]) / 2 + lbl_offset[0]
        my = (src_pt[1] + dst_pt[1]) / 2 + lbl_offset[1]
        ax.text(mx, my, label, ha='center', va='center', fontsize=8,
                color=color, fontweight='bold', zorder=5,
                bbox=dict(facecolor=C_BG, edgecolor='none', pad=1.5))

    # occ ↔ occ : next / prev
    p1 = node_edge_pt('occ', 'top');  p2 = (p1[0] + 0.6, p1[1] + 0.5)
    ax.annotate('', xy=(p2[0] + 0.5, p2[1] - 0.02), xytext=p2,
                arrowprops=dict(arrowstyle='->', color=EDGE_COLORS['next'], lw=1.8,
                                connectionstyle='arc3,rad=-0.5'))
    ax.text(p1[0] + 1.1, p1[1] + 0.55, 'next  /  prev\n(occ[i] ↔ occ[i+1])',
            ha='left', va='center', fontsize=8.5, color=EDGE_COLORS['next'],
            fontweight='bold')

    # occ → chord : instance_of
    draw_arrow(node_edge_pt('occ', 'bottom'), node_edge_pt('chord', 'top'),
               'instance_of', EDGE_COLORS['instance_of'], rad=0.15, lbl_offset=(-0.5, 0))
    # chord → occ : inst_rev
    draw_arrow(node_edge_pt('chord', 'top'), node_edge_pt('occ', 'bottom'),
               'inst_rev', EDGE_COLORS['inst_rev'], rad=-0.15, lbl_offset=(0.5, 0))

    # occ → sec : in_section
    draw_arrow(node_edge_pt('occ', 'right'), node_edge_pt('sec', 'left'),
               'in_section', EDGE_COLORS['in_section'], rad=-0.25, lbl_offset=(0, 0.35))
    # sec → occ : sec_rev
    draw_arrow(node_edge_pt('sec', 'left'), node_edge_pt('occ', 'right'),
               'sec_rev', EDGE_COLORS['sec_rev'], rad=0.25, lbl_offset=(0, -0.35))

    # sec → sec : next_section
    p = node_edge_pt('sec', 'top')
    ax.annotate('', xy=(p[0] + 0.55, p[1] - 0.02), xytext=(p[0] + 0.05, p[1] + 0.35),
                arrowprops=dict(arrowstyle='->', color=EDGE_COLORS['next_section'], lw=1.8,
                                connectionstyle='arc3,rad=-0.6'))
    ax.text(p[0] + 1.3, p[1] + 0.4, 'next_section',
            ha='left', va='center', fontsize=8.5,
            color=EDGE_COLORS['next_section'], fontweight='bold')

    # chord → note : chord_contains
    draw_arrow(node_edge_pt('chord', 'right'), node_edge_pt('note', 'left'),
               'chord_contains\n/ note_in_chord',
               EDGE_COLORS['chord_contains'], rad=0.0, lbl_offset=(0, 0.22))

    # chord → scale_deg : chord_degree
    draw_arrow(node_edge_pt('chord', 'right'), node_edge_pt('scale_deg', 'left'),
               'chord_degree\n/ degree_rev',
               EDGE_COLORS['chord_degree'], rad=0.0, lbl_offset=(0, -0.15))

    # ── Legend: node types ────────────────────────────────────────────────────
    patches = [mpatches.Patch(facecolor=c + '55', edgecolor=c, linewidth=2, label=n)
               for n, c in NODE_COLORS.items()]
    leg = ax.legend(handles=patches, loc='lower left',
                    fontsize=10, framealpha=0.25,
                    facecolor='#111111', edgecolor='#555555',
                    title='Node types', title_fontsize=10)
    leg.get_title().set_color(C_TEXT)
    for t in leg.get_texts(): t.set_color(C_TEXT)

    ax.set_title('MusicHeteroGNN — Graph Schema',
                 fontsize=18, fontweight='bold', color=C_TEXT, pad=8)

    fig.tight_layout(pad=0.5)
    out = f'{RESULTS}/graph_schema.png'
    fig.savefig(out, dpi=180, facecolor=C_BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Instance graph (real song excerpt)
# ══════════════════════════════════════════════════════════════════════════════

def draw_instance():
    from src.dataset import get_splits
    from src.vocab   import VOCAB_SIZE

    # Load a small, well-known-structure song
    train_g, _, _ = get_splits('data/McGill-Billboard', 'data/processed')

    # Pick a graph that renders well:
    #   - 12–20 occ nodes (readable timeline)
    #   - at least 4 unique chord types (not degenerate single-chord songs)
    #   - at least 2 sections
    #   - occ/sec ratio between 2 and 6 (not weirdly many sections)
    target = None
    for g in train_g:
        n_occ   = g['occ'].x.shape[0]
        n_chord = g['chord'].x.shape[0]
        n_sec   = g['sec'].x.shape[0]
        if (18 <= n_occ <= 30
                and n_chord >= 4
                and n_sec >= 2
                and 2.0 <= n_occ / n_sec <= 7.0):
            target = g
            break
    if target is None:
        target = train_g[0]

    g      = target
    N_occ  = g['occ'].x.shape[0]
    N_chord = g['chord'].x.shape[0]
    N_sec   = g['sec'].x.shape[0]
    N_note  = g['note'].x.shape[0]   # always 12
    N_sd    = g['scale_deg'].x.shape[0]  # always 12

    # ── Build networkx graph ─────────────────────────────────────────────────
    G = nx.DiGraph()

    # Node naming
    occ_ids  = [f'occ{i}'   for i in range(N_occ)]
    ch_ids   = [f'ch{i}'    for i in range(N_chord)]
    sec_ids  = [f'sec{i}'   for i in range(N_sec)]
    note_ids = [f'n{i}'     for i in range(N_note)]
    sd_ids   = [f'sd{i}'    for i in range(N_sd)]

    for nid in occ_ids:  G.add_node(nid, ntype='occ')
    for nid in ch_ids:   G.add_node(nid, ntype='chord')
    for nid in sec_ids:  G.add_node(nid, ntype='sec')
    for nid in note_ids: G.add_node(nid, ntype='note')
    for nid in sd_ids:   G.add_node(nid, ntype='scale_deg')

    def add_edges(src_list, dst_list, etype):
        ei = g.edge_index_dict.get(etype)
        if ei is None: return
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            G.add_edge(src_list[s], dst_list[d], etype=etype[1])

    add_edges(occ_ids, occ_ids,   ('occ', 'next', 'occ'))
    add_edges(occ_ids, ch_ids,    ('occ', 'instance_of', 'chord'))
    add_edges(ch_ids,  occ_ids,   ('chord', 'inst_rev', 'occ'))
    add_edges(occ_ids, sec_ids,   ('occ', 'in_section', 'sec'))
    add_edges(sec_ids, occ_ids,   ('sec', 'sec_rev', 'occ'))
    add_edges(sec_ids, sec_ids,   ('sec', 'next_section', 'sec'))
    add_edges(ch_ids,  note_ids,  ('chord', 'chord_contains', 'note'))
    add_edges(note_ids, ch_ids,   ('note', 'note_in_chord', 'chord'))
    add_edges(ch_ids,  sd_ids,    ('chord', 'chord_degree', 'scale_deg'))
    add_edges(sd_ids,  ch_ids,    ('scale_deg', 'degree_rev', 'chord'))

    # ── Custom layout ────────────────────────────────────────────────────────
    pos = {}
    PAD = 0.15

    # occ: horizontal timeline across the middle
    for i, nid in enumerate(occ_ids):
        pos[nid] = (i / max(N_occ - 1, 1) * 10, 4.0)

    # sec: above occ, spread by their associated occ positions
    ei_insec = g.edge_index_dict.get(('occ', 'in_section', 'sec'))
    sec_x_acc = {i: [] for i in range(N_sec)}
    if ei_insec is not None:
        for occ_i, sec_j in zip(ei_insec[0].tolist(), ei_insec[1].tolist()):
            sec_x_acc[sec_j].append(pos[occ_ids[occ_i]][0])
    for i, nid in enumerate(sec_ids):
        xs = sec_x_acc.get(i, [i * 2.0])
        pos[nid] = (np.mean(xs) if xs else i * 2, 6.8)

    # chord: below occ, evenly spaced
    for i, nid in enumerate(ch_ids):
        pos[nid] = (i / max(N_chord - 1, 1) * 10, 2.0)

    # note: bottom-left arc
    for i, nid in enumerate(note_ids):
        angle = np.pi + (i / 12) * np.pi
        pos[nid] = (-1.5 + 1.2 * np.cos(angle), 0.3 + 0.8 * np.sin(angle))

    # scale_deg: bottom-right arc
    for i, nid in enumerate(sd_ids):
        angle = (i / 12) * np.pi
        pos[nid] = (11.5 + 1.2 * np.cos(angle), 0.3 + 0.8 * np.sin(angle))

    # ── Draw ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.axis('off')

    # Draw edges by type
    edge_groups = {}
    for u, v, d in G.edges(data=True):
        et = d['etype']
        edge_groups.setdefault(et, []).append((u, v))

    for etype, edges in edge_groups.items():
        color = EDGE_COLORS.get(etype, '#aaaaaa')
        # Separate self-type edges (occ→occ) to use curved style
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=color, alpha=0.6,
            arrows=True, arrowsize=12,
            width=1.5,
            connectionstyle='arc3,rad=0.15' if etype in ('next', 'next_section') else 'arc3,rad=0.0',
        )

    # Draw nodes by type
    for ntype, color in NODE_COLORS.items():
        nodelist = [n for n, d in G.nodes(data=True) if d['ntype'] == ntype]
        size = {'occ': 800, 'chord': 700, 'sec': 900,
                'note': 400, 'scale_deg': 400}[ntype]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, ax=ax,
                               node_color=color, node_size=size,
                               alpha=0.92, linewidths=2,
                               edgecolors='white')

    # Labels on occ nodes: chord name from labels
    from src.vocab import VOCAB_SIZE as VS
    ROOTS  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    QUALS  = ['maj','min','dim','aug','sus','?']

    def cid_to_str(cid):
        if cid >= VS: return 'N'
        root = cid // 12; qual = (cid % 12) // 2; cx = cid % 2
        return f'{ROOTS[root]}\n{QUALS[qual]}{"7" if cx else ""}'

    ei_inst = g.edge_index_dict.get(('occ', 'instance_of', 'chord'))
    ei_ch   = g.edge_index_dict.get(('chord', 'inst_rev', 'occ'))
    chord_feat = g['chord'].x
    occ_chord_str = {}
    if ei_inst is not None:
        root_ids = chord_feat[:, :12].argmax(1)
        qual_ids = chord_feat[:, 12:18].argmax(1)
        cx_ids   = chord_feat[:, 18:20].argmax(1)
        for occ_i, ch_j in zip(ei_inst[0].tolist(), ei_inst[1].tolist()):
            r = root_ids[ch_j].item(); q = qual_ids[ch_j].item(); c = cx_ids[ch_j].item()
            cid = r * 12 + q * 2 + c
            occ_chord_str[occ_ids[occ_i]] = f'{ROOTS[r]}\n{QUALS[q]}{"7" if c else ""}'

    occ_labels  = {n: occ_chord_str.get(n, '') for n in occ_ids}
    sec_labels  = {f'sec{i}': f'sec{i}' for i in range(N_sec)}
    note_labels = {f'n{i}': ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][i]
                   for i in range(12)}
    sd_labels   = {f'sd{i}': f'°{i}' for i in range(12)}

    all_labels = {**occ_labels, **sec_labels, **note_labels, **sd_labels}
    nx.draw_networkx_labels(G, pos, labels=all_labels, ax=ax,
                            font_size=7, font_color='white', font_weight='bold')

    # ── Legend ────────────────────────────────────────────────────────────────
    node_patches = [mpatches.Patch(facecolor=c, edgecolor='white',
                                   linewidth=1.5, label=n)
                    for n, c in NODE_COLORS.items()]
    edge_patches = [mpatches.Patch(facecolor=EDGE_COLORS[e], label=e)
                    for e in ('next', 'instance_of', 'inst_rev',
                              'in_section', 'sec_rev', 'next_section',
                              'chord_contains', 'chord_degree')]
    leg1 = ax.legend(handles=node_patches, loc='upper left',
                     title='Node types', fontsize=9, title_fontsize=9,
                     facecolor='#111111', edgecolor='#555555', framealpha=0.6)
    leg1.get_title().set_color(C_TEXT)
    for t in leg1.get_texts(): t.set_color(C_TEXT)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=edge_patches, loc='upper right',
                     title='Edge types', fontsize=9, title_fontsize=9,
                     facecolor='#111111', edgecolor='#555555', framealpha=0.6,
                     ncol=2)
    leg2.get_title().set_color(C_TEXT)
    for t in leg2.get_texts(): t.set_color(C_TEXT)

    # Tier labels
    for txt, y in [('sec nodes', 6.8), ('occ nodes  (timeline →)', 4.0),
                   ('chord nodes', 2.0), ('note / scale_deg nodes', 0.3)]:
        ax.text(-2.5, y, txt, ha='left', va='center', fontsize=9,
                color='#aaaaaa', style='italic')

    title = getattr(g, 'song_title', '') or getattr(g, 'song_id', 'song')
    ax.set_title(f'MusicHeteroGNN — Instance Graph\n"{title}"  ({N_occ} occ · {N_chord} chord · {N_sec} sec nodes)',
                 fontsize=15, fontweight='bold', color=C_TEXT)

    fig.tight_layout(pad=0.3)
    out = f'{RESULTS}/graph_instance.png'
    fig.savefig(out, dpi=180, facecolor=C_BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    draw_schema()
    draw_instance()
    print('Done.')
