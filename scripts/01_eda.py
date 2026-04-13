"""
Script 01 — Exploratory Data Analysis

Run from project root:
    python scripts/01_eda.py

Prints:
    - Total songs parsed
    - Average / median chord count per song
    - Section type distribution
    - Top-30 chord types
    - Chord vocab coverage after normalization
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from pathlib import Path

from src.parse import load_all_songs
from src.vocab import normalize_chord_to_id, normalize_section_type, VOCAB_SIZE, N_CHORD_ID

DATA_DIR = 'data/McGill-Billboard'


def main():
    songs = load_all_songs(DATA_DIR)
    print(f'\n=== Dataset Overview ===')
    print(f'Songs parsed:  {len(songs)}')

    chord_counts = [len(s['chords']) for s in songs]
    sec_counts   = [len(s['sections']) for s in songs]

    import statistics
    print(f'Chords/song:   mean={statistics.mean(chord_counts):.1f}, '
          f'median={statistics.median(chord_counts):.0f}, '
          f'min={min(chord_counts)}, max={max(chord_counts)}')
    print(f'Sections/song: mean={statistics.mean(sec_counts):.1f}, '
          f'median={statistics.median(sec_counts):.0f}')

    # Section type distribution
    sec_type_counter: Counter = Counter()
    for song in songs:
        for sec in song['sections']:
            sec_type_counter[normalize_section_type(sec['section_type'])] += 1

    print(f'\n=== Section type distribution ===')
    for stype, cnt in sec_type_counter.most_common():
        print(f'  {stype:<20s} {cnt:5d}')

    # Raw chord string distribution
    raw_chord_counter: Counter = Counter()
    norm_chord_counter: Counter = Counter()
    for song in songs:
        for c in song['chords']:
            raw_chord_counter[c['chord_str']] += 1
            cid = normalize_chord_to_id(c['chord_str'])
            norm_chord_counter[cid] += 1

    print(f'\n=== Top 30 raw chord strings ===')
    for cs, cnt in raw_chord_counter.most_common(30):
        cid = normalize_chord_to_id(cs)
        print(f'  {cs:<25s} {cnt:6d}  →  id {cid}')

    print(f'\n=== Chord vocab after normalization ===')
    print(f'  Unique raw chord strings:  {len(raw_chord_counter)}')
    print(f'  Unique normalized chord IDs: {len(norm_chord_counter)}')
    print(f'  Max possible vocab IDs:    {VOCAB_SIZE + 1}  (0..{VOCAB_SIZE})')
    n_coverage = sum(1 for cid in norm_chord_counter if cid != N_CHORD_ID)
    print(f'  Non-N IDs used:            {n_coverage}')

    # Total training examples
    total_occ = sum(len(s['chords']) for s in songs)
    print(f'\n=== Training data volume ===')
    print(f'  Total chord occurrences: {total_occ}')
    print(f'  Total next-chord pairs:  {total_occ - len(songs)}')


if __name__ == '__main__':
    main()
