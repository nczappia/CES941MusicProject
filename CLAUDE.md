# CES 941 — Machine Learning on Graphs: Final Project

## Course Info
- **Instructors:** Jiliang Tang, Haoyu Han, Xinnan Dai, Zhikai Chen
- **Time:** Tue/Thu 10:20–11:40 AM, EB 2320
- **Presentations:** April 14 & 16, 2026

## Project Expectations
Projects may be one of:
1. Novel research on graph learning methods
2. Empirical evaluation of existing models
3. Applications of graph deep learning to real-world problems
4. Systems involving graph–LLM integration

This project falls under **categories 1 and 3**: a novel heterogeneous GNN applied to music chord prediction.

## Our Project
**Task:** Next-chord prediction on the McGill Billboard dataset using a heterogeneous GNN (MusicHeteroGNN).

**Novelty claim:** Modeling chord progressions as a heterogeneous graph (occ, chord, sec node types with 7 edge relation types) rather than a flat sequence, and showing via ablation which structural edges matter.

## Critical Issue: Causal Leakage via `prev` Edges
The `prev` edge type (occ[i+1] → occ[i]) creates a data leakage path in the 2-layer GNN:
- Layer 1: chord type identity flows `chord[i+1] →(inst_rev)→ occ[i+1]`
- Layer 2: that enriched embedding flows `occ[i+1] →(prev)→ occ[i]`
- Result: when predicting chord_id[i+1] from occ[i], the model has already seen it

This explains the ~99.8% test accuracy (vs LSTM 45.2%, Markov 23.5%).
The ablation confirms it: removing `instance_of` drops to 7.5% because without chord features
in occ embeddings, the `prev` path can't carry chord identity backward.

**Fix:** Train a "causal" model with `use_seq_edges=True` but removing `prev` edges only.
This requires a minor change to `src/graph.py` and `src/model.py`.

## Results Summary
| Model                                  | Top-1 Acc | Top-5 Acc | Cross-Entropy |
|----------------------------------------|-----------|-----------|---------------|
| Markov                                 | 23.5%     | 60.6%     | 4.58          |
| LSTM                                   | 45.2%     | 83.1%     | 1.98          |
| Causal GNN v1 (no prev)                | ~46%      | ~82%      | ~1.96         |
| Causal GNN v2 (chord-in-occ, 3 layers) | 60.8%     | 89.0%     | 1.44          |
| GNN (full, leaky)                      | 99.8%     | 99.9%     | 0.015         |
| GNN no seq edges                       | 25.4%     | 64.3%     | 2.83          |
| GNN no inst_of                         | 7.5%      | 33.8%     | 3.73          |
| GNN no sec edges                       | 99.9%     | 99.9%     | 0.012         |

Causal GNN v2 checkpoint: `results/causal_gnn_best.pt`
- `use_prev_edges=False`, `use_chord_in_occ=True`, `num_layers=3`, `hidden_dim=128`

## Pipeline
1. `scripts/01_eda.py` — EDA
2. `scripts/02_baselines.py` — Markov + LSTM baselines
3. `scripts/03_train_gnn.py` — Train full HeteroGNN (60 epochs)
4. `scripts/04_ablation.py` — Ablation study
5. `scripts/05_visualize.py` — Final plots (UMAP, section accuracy, ablation bars)
6. `scripts/06_train_causal_gnn.py` — Train causal GNN v2 (no prev edges, chord-in-occ, 3 layers)
7. `scripts/07_era_umap.py` — Era-proxy UMAP (chart decade coloring). Output: `results/era_umap.png`
8. `scripts/08_genre_analysis.py` — Genre UMAP + chord heatmap + transition bars
9. `scripts/09_gat_attention.py` — Train causal GAT model + extract/visualize attention weights
10. `scripts/10_multitask_genre.py` — Multi-task GNN: joint chord + genre prediction

## Completed Work

### Genre Labels
`data/genre_labels.json` — 890 songs, 93.8% match rate via MusicBrainz.
Coarse genre buckets: rock(459), pop(169), other(66), soul_r&b(63), disco_dance(32),
country(31), folk(24), jazz(23), blues(19). hip_hop(4) merged into 'other' in multitask script.

### Genre Analysis (completed 2026-04-13)
- `results/genre_umap.png` — UMAP by genre: genres broadly mixed, chord embeddings don't encode genre
- `results/genre_chord_heatmap.png` — hip_hop concentrates on minor chords; blues distinct; rock/pop overlap
- `results/genre_progression_bars.png` — top bigram transitions per genre

### Pre-computed Embedding Assets
- `results/era_umap.png` — UMAP by decade: eras mixed, chord embeddings don't encode era
- `results/song_embeddings.npy` — [890, 128] mean-pooled occ embeddings from causal GNN v2
- `results/song_ids.json` — ordered song IDs matching embeddings
- `results/song_embeddings_meta.json` — per-song UMAP coords + era label

### GAT Attention (completed 2026-04-13)
`src/model.py` — `MusicHeteroGNN` accepts `use_attention=True, gat_heads=4` (SAGEConv → GATConv).
- `results/gat_best.pt` — trained checkpoint (61.3% top-1, 90.1% top-5)
- `results/gat_training_curves.png`
- `results/gat_attention_by_relation.png` — two-tier result: inst_rev + next + next_section + sec_rev
  all get ~1.0 attention; in_section + instance_of get ~0.13. Both chord identity and sequence matter.
- `results/gat_attention_heatmap.png` — sparse, structured attention over chord transitions

### Bug Fixes Applied
- `src/train.py:_get_occ_embeddings` — fixed chord feature injection for `use_chord_in_occ=True`
  and `use_prev_edges` edge filtering (both were broken for causal v2).

## Multi-task Genre Classification (in progress as of 2026-04-13)

`src/model.py` updated — `MusicHeteroGNN` now accepts `num_genres=N` (default 0).
When >0, adds `genre_head = Linear(128 → N)` and a `forward_with_genre()` method:
- Runs full GNN forward pass
- Scatter-means occ embeddings per song (via PyG batch pointer)
- Returns `(chord_logits [N_occ, C], genre_logits [B, num_genres])`

`scripts/10_multitask_genre.py` — trains causal v2 config + genre head jointly.
- Loss = chord_CE + 0.5 × genre_CE
- Genre loss masked for songs without MusicBrainz labels
- 9 genre classes (hip_hop merged into other)

**Training is currently running in tmux session `multitask`.**
- Reattach: `tmux attach -t multitask`
- If session gone: `source venv/bin/activate && python scripts/10_multitask_genre.py`
- Expected outputs when done:
  - `results/multitask_best.pt`
  - `results/multitask_history.json`
  - `results/multitask_training_curves.png` — 3 panels: chord CE, chord top-1, genre acc
  - `results/multitask_genre_umap.png` — UMAP with genre-supervised embeddings (should cluster better)
  - `results/multitask_genre_acc.json` — overall + per-genre test accuracy

## Remaining / Stretch
- **Contrastive learning** — GraphCL-style augmentation for genre-separable embeddings without labels
