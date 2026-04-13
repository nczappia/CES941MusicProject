"""
Chord vocabulary and normalization for the McGill Billboard dataset.

Chord ID scheme:
    vocab_id = root * 12 + quality * 2 + complexity
    root: 0-11 (C=0, C#/Db=1, ..., B=11)
    quality: 0=maj, 1=min, 2=dim, 3=aug, 4=sus, 5=other
    complexity: 0=triad, 1=seventh_or_more
    Total: 12 * 6 * 2 = 144 chord types
    N_CHORD_ID = 144  (no-chord / unknown)
"""

import re
import torch
import torch.nn.functional as F

VOCAB_SIZE = 12 * 6 * 2   # 144
N_CHORD_ID = VOCAB_SIZE    # 144  — "no chord" / silence sentinel

ROOT_NAMES = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11,
}

QUALITY_NAMES = ['maj', 'min', 'dim', 'aug', 'sus', 'other']
COMPLEXITY_NAMES = ['triad', 'seventh_or_more']

SECTION_TYPES = [
    'intro', 'verse', 'chorus', 'bridge', 'prechorus',
    'outro', 'instrumental', 'interlude', 'solo', 'trans', 'other',
]
SECTION_TYPE_TO_ID = {s: i for i, s in enumerate(SECTION_TYPES)}
NUM_SECTION_TYPES = len(SECTION_TYPES)  # 11

_SECTION_NORM = {
    'pre-chorus': 'prechorus',
    'pre chorus': 'prechorus',
    'precorus': 'prechorus',
    'instumental': 'instrumental',
    'instrument': 'instrumental',
    'fade': 'outro',
    'fadeout': 'outro',
    'coda': 'outro',
    'modulation': 'trans',
    'silence': 'other',
    'unknown': 'other',
}


def normalize_section_type(raw: str) -> str:
    s = raw.strip().lower()
    s = _SECTION_NORM.get(s, s)
    return s if s in SECTION_TYPE_TO_ID else 'other'


def section_type_to_id(raw: str) -> int:
    return SECTION_TYPE_TO_ID[normalize_section_type(raw)]


def parse_root(root_str: str) -> int:
    root_str = root_str.strip()
    if root_str in ROOT_NAMES:
        return ROOT_NAMES[root_str]
    if len(root_str) >= 2 and root_str[:2] in ROOT_NAMES:
        return ROOT_NAMES[root_str[:2]]
    if root_str[:1] in ROOT_NAMES:
        return ROOT_NAMES[root_str[:1]]
    return 0


def parse_quality(qual_str: str) -> int:
    """Map quality token to 0=maj,1=min,2=dim,3=aug,4=sus,5=other."""
    if not qual_str:
        return 0
    q = qual_str.lower()
    if 'hdim' in q:
        return 2
    if q.startswith('min'):
        return 1
    if q.startswith('maj'):
        return 0
    if q.startswith('dim'):
        return 2
    if q.startswith('aug'):
        return 3
    if q.startswith('sus'):
        return 4
    if q == '5':
        return 5
    # Dominant / plain numeric (7, 6, 9, 11, 13) → major-based
    if re.match(r'^\d+$', q):
        return 0
    return 0


def parse_complexity(qual_str: str, paren_ext: str = '') -> int:
    """0=triad, 1=seventh_or_more (any extension beyond a plain triad)."""
    combined = (qual_str + ' ' + paren_ext).lower()
    for ind in ('6', '7', '9', '11', '13'):
        if ind in combined:
            return 1
    return 0


def normalize_chord(chord_str: str):
    """
    Returns (root_id, quality_id, complexity_id).
    Unknown / no-chord symbols return (0, 5, 0) — mapped to N_CHORD_ID via chord_to_id.
    """
    chord_str = chord_str.strip()
    if chord_str in ('N', 'X', ''):
        return (0, 5, 0)

    if ':' not in chord_str:
        root_str = re.split(r'[/(\s]', chord_str)[0]
        return (parse_root(root_str), 0, 0)

    root_str, qual_raw = chord_str.split(':', 1)

    # Remove bass note
    qual_raw = re.sub(r'/.*$', '', qual_raw)

    # Extract parenthetical extension
    paren_ext = ''
    m = re.search(r'\(([^)]*)\)', qual_raw)
    if m:
        paren_ext = m.group(1)
        qual_raw = qual_raw[:m.start()] + qual_raw[m.end():]
    qual_raw = qual_raw.strip()

    root_id = parse_root(root_str)
    quality_id = parse_quality(qual_raw)
    complexity_id = parse_complexity(qual_raw, paren_ext)

    return (root_id, quality_id, complexity_id)


def chord_to_id(root: int, quality: int, complexity: int) -> int:
    return root * 12 + quality * 2 + complexity


def normalize_chord_to_id(chord_str: str) -> int:
    """Full pipeline: chord string → integer vocab id (0..144)."""
    chord_str = chord_str.strip()
    if chord_str in ('N', 'X', ''):
        return N_CHORD_ID
    root, quality, complexity = normalize_chord(chord_str)
    cid = chord_to_id(root, quality, complexity)
    # N and X map here as well if they slipped through
    if chord_str in ('N', 'X'):
        return N_CHORD_ID
    return cid


def chord_id_to_features(chord_id: int) -> torch.Tensor:
    """chord_id → 20-dim feature vector [root_12 | quality_6 | complexity_2]."""
    if chord_id == N_CHORD_ID:
        return torch.zeros(20)
    root = chord_id // 12
    quality = (chord_id % 12) // 2
    complexity = chord_id % 2
    r = F.one_hot(torch.tensor(root), 12).float()
    q = F.one_hot(torch.tensor(quality), 6).float()
    c = F.one_hot(torch.tensor(complexity), 2).float()
    return torch.cat([r, q, c])
