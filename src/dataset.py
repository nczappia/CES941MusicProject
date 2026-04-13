"""
Dataset: load all songs, build heterographs, cache to disk, return splits.

Usage
-----
    from src.dataset import load_dataset, get_splits

    train_data, val_data, test_data = get_splits(data_dir, processed_dir)
    # each is a list of HeteroData objects
"""

import os
import pickle
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch_geometric.data import HeteroData

from .parse import load_all_songs
from .graph import build_song_heterograph


CACHE_FILE = 'graphs.pkl'


def build_and_cache(data_dir: str, processed_dir: str, force: bool = False) -> List[HeteroData]:
    """Parse all songs, build graphs, pickle to processed_dir/graphs.pkl."""
    cache_path = Path(processed_dir) / CACHE_FILE
    if cache_path.exists() and not force:
        print(f'Loading cached graphs from {cache_path}')
        with open(cache_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f'  Loaded {len(graphs)} graphs.')
        return graphs

    print(f'Parsing songs from {data_dir} ...')
    songs = load_all_songs(data_dir)
    print(f'  Parsed {len(songs)} songs.')

    graphs = []
    skipped = 0
    for song in songs:
        try:
            g = build_song_heterograph(song)
            graphs.append(g)
        except Exception as e:
            skipped += 1

    print(f'  Built {len(graphs)} graphs ({skipped} skipped on error).')
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(graphs, f)
    print(f'  Saved to {cache_path}')
    return graphs


def get_splits(
    data_dir: str,
    processed_dir: str,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    force_rebuild: bool = False,
) -> Tuple[List[HeteroData], List[HeteroData], List[HeteroData]]:
    """Return (train, val, test) lists of HeteroData, split at song level."""
    graphs = build_and_cache(data_dir, processed_dir, force=force_rebuild)

    rng = random.Random(seed)
    indices = list(range(len(graphs)))
    rng.shuffle(indices)

    n_train = int(len(indices) * train_frac)
    n_val   = int(len(indices) * val_frac)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train = [graphs[i] for i in train_idx]
    val   = [graphs[i] for i in val_idx]
    test  = [graphs[i] for i in test_idx]

    print(f'Split: {len(train)} train / {len(val)} val / {len(test)} test songs')
    return train, val, test
