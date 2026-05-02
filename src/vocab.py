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


# ── Krumhansl-Schmuckler key detection ───────────────────────────────────────

# Tonal hierarchy profiles (Krumhansl & Schmuckler 1990)
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _pearson(x, y):
    import math
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    den = math.sqrt(sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y))
    return num / den if den > 0 else 0.0


def tonic_to_root(tonic_str: str) -> int:
    """Parse the tonic string from the annotation header (e.g. 'C', 'F#', 'Bb') → root int 0-11."""
    tonic_str = tonic_str.strip()
    if tonic_str in ROOT_NAMES:
        return ROOT_NAMES[tonic_str]
    # Try first two chars then first char
    if len(tonic_str) >= 2 and tonic_str[:2] in ROOT_NAMES:
        return ROOT_NAMES[tonic_str[:2]]
    if tonic_str[:1] in ROOT_NAMES:
        return ROOT_NAMES[tonic_str[:1]]
    return 0   # fallback to C


def detect_song_key(chord_ids: list) -> tuple:
    """
    Estimate the key of a song from its chord ID sequence.
    Returns (key_root, mode) where key_root ∈ 0-11 and mode ∈ {0=minor, 1=major}.
    Falls back to (0, 1) = C major for silent/empty songs.
    """
    # Build pitch class histogram from chords in song
    pc_hist = [0.0] * 12
    for cid in chord_ids:
        for pc in chord_id_to_pitch_classes(cid):
            pc_hist[pc] += 1.0

    if sum(pc_hist) == 0:
        return (0, 1)

    best_r, best_key, best_mode = -2.0, 0, 1
    for root in range(12):
        # Rotate profiles to this root
        maj = [_KS_MAJOR[(i - root) % 12] for i in range(12)]
        mn  = [_KS_MINOR[(i - root) % 12] for i in range(12)]
        r_maj = _pearson(pc_hist, maj)
        r_min = _pearson(pc_hist, mn)
        if r_maj > best_r:
            best_r, best_key, best_mode = r_maj, root, 1
        if r_min > best_r:
            best_r, best_key, best_mode = r_min, root, 0

    return (best_key, best_mode)


def transpose_chord_id(chord_id: int, semitones: int) -> int:
    """Transpose a chord ID by shifting its root by `semitones` (mod 12). N stays N."""
    if chord_id == N_CHORD_ID:
        return chord_id
    root      = chord_id // 12
    remainder = chord_id % 12   # quality*2 + complexity
    new_root  = (root + semitones) % 12
    return new_root * 12 + remainder


# ── Interval sets (semitones from root) for each (quality, complexity) combination.
# Used to derive the pitch classes present in a chord.
_QUALITY_INTERVALS = {
    (0, 0): [0, 4, 7],           # major triad
    (0, 1): [0, 4, 7, 10],       # dominant 7th (covers maj7 approx)
    (1, 0): [0, 3, 7],           # minor triad
    (1, 1): [0, 3, 7, 10],       # minor 7th
    (2, 0): [0, 3, 6],           # diminished triad
    (2, 1): [0, 3, 6, 9],        # diminished 7th / half-dim
    (3, 0): [0, 4, 8],           # augmented
    (3, 1): [0, 4, 8, 10],       # augmented 7th
    (4, 0): [0, 5, 7],           # sus4
    (4, 1): [0, 5, 7, 10],       # sus4 + 7
    (5, 0): [],                  # other / no-chord
    (5, 1): [],
}


def chord_id_to_pitch_classes(chord_id: int) -> list:
    """
    Returns the list of pitch classes (0-11) present in a chord.
    No-chord (N_CHORD_ID) returns an empty list.
    """
    if chord_id == N_CHORD_ID:
        return []
    root    = chord_id // 12
    quality = (chord_id % 12) // 2
    complexity = chord_id % 2
    intervals = _QUALITY_INTERVALS.get((quality, complexity), [0, 4, 7])
    return [(root + interval) % 12 for interval in intervals]


# ── Metre encoding ────────────────────────────────────────────────────────────

METRE_LABELS = ['4/4', '12/8', '3/4', '6/8', 'other']
METRE_DIM    = len(METRE_LABELS)   # 5


def metre_to_onehot(metre_str: str) -> torch.Tensor:
    """Encode metre string as 5-dim one-hot. Unknown metres → 'other'."""
    metre_str = metre_str.strip()
    idx = METRE_LABELS.index(metre_str) if metre_str in METRE_LABELS else METRE_LABELS.index('other')
    return F.one_hot(torch.tensor(idx), METRE_DIM).float()


# ── Chord extension features ──────────────────────────────────────────────────
# 5 binary flags parsed from the raw chord string (cannot be recovered from chord_id alone)
# [is_dom7, is_maj7, is_min7, is_sus, is_extended]
CHORD_EXT_DIM = 5


def chord_str_to_extension_features(chord_str: str) -> torch.Tensor:
    """
    Parse raw chord string → 5-dim extension feature vector.
    Flags: [is_dom7, is_maj7, is_min7, is_sus, is_extended(9th+)]
    """
    if chord_str in ('N', 'X', '', 'silence'):
        return torch.zeros(CHORD_EXT_DIM)

    s = chord_str.lower()
    qual = s.split(':', 1)[1] if ':' in s else ''
    # Strip bass note and parentheticals for cleaner matching
    qual = re.sub(r'/.*$', '', qual)
    qual_clean = re.sub(r'\([^)]*\)', '', qual).strip()
    paren = re.search(r'\(([^)]*)\)', qual)
    paren_str = paren.group(1).lower() if paren else ''
    combined = qual_clean + ' ' + paren_str

    is_maj7     = int(bool(re.search(r'maj7|maj9|maj11|maj13', combined)))
    is_min7     = int(bool(re.search(r'min7|min9|min11|min13|m7', combined)))
    is_dom7     = int(bool(
        re.search(r'^7|^9|^11|^13', qual_clean.strip()) or
        (re.search(r'7', combined) and not is_maj7 and not is_min7)
    ))
    is_sus      = int(bool(re.search(r'sus', combined)))
    is_extended = int(bool(re.search(r'9|11|13', combined)))

    return torch.tensor([is_dom7, is_maj7, is_min7, is_sus, is_extended], dtype=torch.float)


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
