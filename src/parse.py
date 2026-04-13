"""
Parser for McGill Billboard salami_chords.txt files.

Each file produces a dict:
    {
        'title':    str,
        'artist':   str,
        'metre':    str,
        'tonic':    str,
        'chords':   [{'start', 'end', 'chord_str', 'section_label',
                      'section_type', 'section_idx'}, ...],
        'sections': [{'start', 'end', 'section_type', 'section_label',
                      'section_idx'}, ...],
    }
Returns None for files that parse to 0 chord events.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_section(content: str) -> Tuple[Optional[str], str, str]:
    """
    Strip instrument annotations and detect an optional section label.

    Returns (section_label | None, section_type_str, bar_content_str).

    Section label format at line start: 'A, intro, | ...'
    Lines that are bar-only look like:            '| A:min | C:maj |'
    Continuation lines begin with:               '-> | ...'
    """
    # Remove instrument annotations from end of line.
    # They appear as:  ', (voice'  or  'voice)'  after the final bar.
    content = re.sub(r',\s*\([^)]*$', '', content).strip()
    content = re.sub(r',\s*[a-zA-Z][a-zA-Z\s]*\)\s*$', '', content).strip()

    # Section label: single uppercase letter, comma, type phrase, comma, rest
    m = re.match(r'^([A-Z])\s*,\s*([^,|]+?)\s*,\s*(.*)', content, re.DOTALL)
    if m:
        label = m.group(1)
        sec_type = m.group(2).strip().lower()
        rest = m.group(3).strip()
        return label, sec_type, rest

    # Strip leading continuation marker
    content = re.sub(r'^->\s*', '', content).strip()
    return None, '', content


def _parse_bars(bar_str: str) -> List[str]:
    """
    Extract an ordered list of chord-string tokens from a bar-string.

    Handles:
        - Multiple bars:   '| A:min | C:maj |'
        - Beat dots:       '| F:maj . G:maj |'
        - Repeat marker:   '| C:maj | x4'   → 4 × C:maj
        - N (no chord):    excluded
    """
    if not bar_str:
        return []

    # Detect trailing repeat marker:  '| ... | xN'
    repeat = 1
    m = re.search(r'\|\s*x(\d+)\s*$', bar_str)
    if m:
        repeat = int(m.group(1))
        bar_str = bar_str[:m.start()].rstrip() + '|'

    # Split by '|'; odd-indexed segments are bar contents
    parts = bar_str.split('|')
    tokens: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:          # outside bars
            continue
        part = part.strip()
        if not part:
            continue
        for tok in part.split():
            if tok == '.':
                continue
            # Skip known non-chord tokens
            if tok in ('N', 'X', 'silence', 'end', '->'):
                continue
            # Must start with a root note letter A-G (uppercase or lowercase)
            if re.match(r'^[A-Ga-g][b#]?', tok):
                tokens.append(tok)

    return tokens * repeat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_salami_chords(filepath: str) -> Optional[Dict]:
    """Parse one salami_chords.txt file; returns None on failure."""
    path = Path(filepath)
    if not path.exists():
        return None

    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()

    # ---- parse header ----
    meta = {'title': '', 'artist': '', 'metre': '4/4', 'tonic': 'C'}
    content_lines: List[str] = []
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            m = re.match(r'#\s*(\w+):\s*(.*)', line)
            if m:
                key, val = m.group(1).lower(), m.group(2).strip()
                if key in meta:
                    meta[key] = val
        elif line:
            content_lines.append(line)

    # ---- parse timed content lines ----
    timed: List[Tuple[float, str]] = []
    for line in content_lines:
        parts = line.split('\t', 1)
        if len(parts) != 2:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        timed.append((t, parts[1].strip()))

    if not timed:
        return None

    # ---- collect raw chord events ----
    # raw_events: (start_time, chord_str, sec_label, sec_type, section_idx)
    raw_events: List[Tuple] = []
    section_changes: List[Tuple] = []   # (time, label, type, idx)

    current_sec_label = 'A'
    current_sec_type  = 'unknown'
    section_idx       = -1

    for i, (t, content) in enumerate(timed):
        # Skip silence / end markers
        if re.match(r'^(silence|end)\b', content):
            continue

        # Compute line duration: time until the next non-silence line
        next_t = None
        for j in range(i + 1, len(timed)):
            nt, nc = timed[j]
            if not re.match(r'^(silence|end)\b', nc):
                next_t = nt
                break
        if next_t is None:
            for j in range(i + 1, len(timed)):
                nt, nc = timed[j]
                if re.match(r'^(silence|end)\b', nc):
                    next_t = nt
                    break
        if next_t is None:
            next_t = t + 4.0      # fallback

        line_dur = max(next_t - t, 0.01)

        # Extract section and bar content
        sec_label, sec_type, bar_str = _extract_section(content)

        if sec_label is not None:
            current_sec_label = sec_label
            current_sec_type  = sec_type
            section_idx += 1
            section_changes.append((t, sec_label, sec_type, section_idx))

        chord_tokens = _parse_bars(bar_str)
        if not chord_tokens:
            continue

        slot_dur = line_dur / len(chord_tokens)
        for k, cstr in enumerate(chord_tokens):
            raw_events.append((
                t + k * slot_dur,
                cstr,
                current_sec_label,
                current_sec_type,
                max(section_idx, 0),
            ))

    if not raw_events:
        return None

    raw_events.sort(key=lambda x: x[0])

    # ---- build chord list with end times ----
    chords: List[Dict] = []
    for i, (start, cstr, slabel, stype, sidx) in enumerate(raw_events):
        if i + 1 < len(raw_events):
            end = raw_events[i + 1][0]
        else:
            # last chord: duration = mean of all previous durations
            if len(raw_events) > 1:
                mean_dur = (raw_events[-1][0] - raw_events[0][0]) / (len(raw_events) - 1)
            else:
                mean_dur = 0.5
            end = start + mean_dur
        chords.append({
            'start':         start,
            'end':           end,
            'chord_str':     cstr,
            'section_label': slabel,
            'section_type':  stype,
            'section_idx':   sidx,
        })

    # ---- build section list ----
    if section_changes:
        sections = []
        for i, (t, slabel, stype, sidx) in enumerate(section_changes):
            if i + 1 < len(section_changes):
                end_t = section_changes[i + 1][0]
            else:
                end_t = raw_events[-1][0]
            sections.append({
                'start':         t,
                'end':           end_t,
                'section_type':  stype,
                'section_label': slabel,
                'section_idx':   sidx,
            })
    else:
        # No section labels — single dummy section
        sections = [{
            'start':         raw_events[0][0],
            'end':           raw_events[-1][0],
            'section_type':  'unknown',
            'section_label': 'A',
            'section_idx':   0,
        }]
        for chord in chords:
            chord['section_idx'] = 0

    return {
        'title':    meta['title'],
        'artist':   meta['artist'],
        'metre':    meta['metre'],
        'tonic':    meta['tonic'],
        'chords':   chords,
        'sections': sections,
    }


def load_all_songs(data_dir: str) -> List[Dict]:
    """
    Walk data_dir for salami_chords.txt files and parse all of them.
    Returns list of parsed song dicts (failures silently skipped).
    """
    songs = []
    for p in sorted(Path(data_dir).rglob('salami_chords.txt')):
        song = parse_salami_chords(str(p))
        if song and len(song['chords']) >= 5:
            song['path'] = str(p)
            song['song_id'] = p.parent.name
            songs.append(song)
    return songs
