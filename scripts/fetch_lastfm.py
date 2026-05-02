"""
Script: fetch_lastfm.py
Fetch Last.fm artist + track tags for all 890 songs and use them to:
  1. Fill in genre labels for the 55 not_found songs
  2. Disambiguate the 66 'other' catch-all songs
  3. Validate / improve existing coarse labels

Last.fm API is free. Get a key at: https://www.last.fm/api/account/create
Set it as:
    export LASTFM_API_KEY=your_key_here
or paste it directly into API_KEY below.

Strategy:
  - Query track.getTopTags (title + artist) first
  - Fall back to artist.getTopTags if track returns < 3 tags
  - Combine tag weights, map to coarse genre buckets
  - Pick the highest-scoring bucket with weight > threshold

Outputs:
  data/lastfm_tags.json       — raw tags per song
  data/genre_labels.json      — updated in-place with improved coarse labels
  results/lastfm_fetch.log

Run from project root:
    source venv/bin/activate && python scripts/fetch_lastfm.py
"""

import sys, os, json, time, re, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from collections import defaultdict

API_KEY      = os.environ.get('LASTFM_API_KEY', '')
GENRE_PATH   = 'data/genre_labels.json'
TAGS_PATH    = 'data/lastfm_tags.json'
LOG_PATH     = 'results/lastfm_fetch.log'
BASE_URL     = 'https://ws.audioscrobbler.com/2.0/'
RATE_SLEEP   = 0.22   # ~4.5 req/s, safely under 5/s limit

# ── Tag → coarse genre mapping ────────────────────────────────────────────────
# Longer/more specific tags listed first so they match before shorter substrings

TAG_MAP = {
    'hip_hop':    {'hip-hop', 'hip hop', 'rap', 'gangsta rap', 'trap',
                   'east coast rap', 'west coast rap', 'conscious rap'},
    'disco_dance':{'disco', 'dance', 'eurodisco', 'hi-nrg', 'dance-pop',
                   'euro disco', 'new wave dance'},
    'blues':      {'blues', 'electric blues', 'delta blues', 'chicago blues',
                   'blues rock', 'texas blues', 'acoustic blues', 'jump blues'},
    'jazz':       {'jazz', 'jazz fusion', 'bebop', 'swing', 'smooth jazz',
                   'cool jazz', 'bossa nova', 'latin jazz', 'big band',
                   'jazz blues', 'acid jazz', 'free jazz'},
    'folk':       {'folk', 'folk rock', 'singer-songwriter', 'acoustic',
                   'traditional folk', 'contemporary folk', 'celtic folk',
                   'british folk', 'americana folk', 'indie folk'},
    'country':    {'country', 'country rock', 'outlaw country', 'bluegrass',
                   'americana', 'honky tonk', 'country pop', 'nashville sound',
                   'western', 'cowboy', 'alt-country'},
    'soul_r&b':   {'soul', 'r&b', 'rnb', 'rhythm and blues', 'funk',
                   'motown', 'neo soul', 'contemporary r&b', 'new jack swing',
                   'quiet storm', 'northern soul', 'southern soul', 'gospel'},
    'pop':        {'pop', 'synth-pop', 'teen pop', 'dance pop', 'electropop',
                   'bubblegum', 'adult contemporary', 'easy listening',
                   'soft pop', 'power pop', 'baroque pop', 'chamber pop'},
    'rock':       {'rock', 'classic rock', 'alternative', 'alternative rock',
                   'hard rock', 'punk', 'punk rock', 'indie rock', 'grunge',
                   'heavy metal', 'metal', 'progressive rock', 'psychedelic',
                   'soft rock', 'garage rock', 'glam rock', 'new wave',
                   'post-punk', 'britpop', 'art rock', 'arena rock',
                   'southern rock', 'heartland rock', 'rockabilly'},
}

# Build reverse lookup: normalised tag string → coarse genre
_TAG_LOOKUP: dict[str, str] = {}
for genre, tags in TAG_MAP.items():
    for t in tags:
        _TAG_LOOKUP[t.lower()] = genre


def coarse_from_tags(tag_list: list[dict]) -> tuple[str, float]:
    """
    tag_list: [{'name': str, 'count': int}, ...]
    Returns (coarse_genre, confidence_score) or ('other', 0).
    """
    scores: dict[str, float] = defaultdict(float)
    total_weight = sum(int(t.get('count', 0)) for t in tag_list) or 1

    for t in tag_list:
        name   = t.get('name', '').lower().strip()
        weight = int(t.get('count', 0)) / total_weight

        # Direct lookup
        if name in _TAG_LOOKUP:
            scores[_TAG_LOOKUP[name]] += weight
            continue

        # Substring match (e.g. "indie rock" matches 'rock')
        for tag_key, genre in _TAG_LOOKUP.items():
            if tag_key in name or name in tag_key:
                scores[genre] += weight * 0.5
                break

    if not scores:
        return 'other', 0.0
    best = max(scores, key=scores.__getitem__)
    return best, round(scores[best], 4)


# ── API helpers ───────────────────────────────────────────────────────────────

def _call(method: str, params: dict) -> dict:
    params.update({'method': method, 'api_key': API_KEY, 'format': 'json'})
    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {'error': str(e)}


def get_track_tags(title: str, artist: str) -> list[dict]:
    data = _call('track.getTopTags', {'track': title, 'artist': artist, 'autocorrect': 1})
    try:
        return data['toptags']['tag']
    except (KeyError, TypeError):
        return []


def get_artist_tags(artist: str) -> list[dict]:
    data = _call('artist.getTopTags', {'artist': artist, 'autocorrect': 1})
    try:
        return data['toptags']['tag']
    except (KeyError, TypeError):
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def clean_artist(artist: str) -> str:
    """Strip featured-artist suffixes that confuse Last.fm."""
    artist = re.split(r'\s+(?:feat|ft|featuring|with|&|and)\s+', artist, flags=re.I)[0]
    return artist.strip().strip('"\'')


def main():
    if not API_KEY:
        print('ERROR: No Last.fm API key found.')
        print('  Get one free at: https://www.last.fm/api/account/create')
        print('  Then run:  export LASTFM_API_KEY=your_key && python scripts/fetch_lastfm.py')
        sys.exit(1)

    Path('results').mkdir(exist_ok=True)
    open(LOG_PATH, 'w').close()

    with open(GENRE_PATH) as f:
        genre_labels = json.load(f)

    # Load existing Last.fm tags cache (resumable)
    if os.path.exists(TAGS_PATH):
        with open(TAGS_PATH) as f:
            tags_cache = json.load(f)
        log(f'Resuming — {len(tags_cache)} songs already fetched')
    else:
        tags_cache = {}

    songs = [(sid, v) for sid, v in genre_labels.items() if sid not in tags_cache]
    log(f'Fetching Last.fm tags for {len(songs)} songs...\n')

    for i, (sid, info) in enumerate(songs):
        title  = info.get('title', '')
        artist = clean_artist(info.get('artist', ''))

        if not title or not artist:
            tags_cache[sid] = {'track_tags': [], 'artist_tags': [], 'coarse': 'other', 'conf': 0}
            continue

        track_tags  = get_track_tags(title, artist)
        time.sleep(RATE_SLEEP)

        # Fall back to artist tags if track returns fewer than 3 useful tags
        if len(track_tags) < 3:
            artist_tags = get_artist_tags(artist)
            time.sleep(RATE_SLEEP)
        else:
            artist_tags = []

        # Merge: track tags get 2× weight via duplication
        combined = track_tags + track_tags + artist_tags
        coarse, conf = coarse_from_tags(combined)

        tags_cache[sid] = {
            'track_tags':  [t['name'] for t in track_tags[:10]],
            'artist_tags': [t['name'] for t in artist_tags[:10]],
            'coarse':      coarse,
            'conf':        conf,
        }

        status = f'{coarse} ({conf:.2f})'
        top_tags = [t['name'] for t in (track_tags or artist_tags)[:4]]
        log(f'[{i+1}/{len(songs)}] {artist} — {title}  →  {status}  | tags: {top_tags}')

        # Save cache every 20 songs
        if (i + 1) % 20 == 0:
            with open(TAGS_PATH, 'w') as f:
                json.dump(tags_cache, f, indent=2)

    with open(TAGS_PATH, 'w') as f:
        json.dump(tags_cache, f, indent=2)
    log(f'\nSaved raw tags to {TAGS_PATH}')

    # ── Update genre_labels.json ──────────────────────────────────────────────
    updated = improved = 0
    CONF_THRESHOLD = 0.08   # minimum confidence to override 'other'

    for sid, lfm in tags_cache.items():
        if sid not in genre_labels:
            continue

        current  = genre_labels[sid].get('coarse', 'other')
        new_coarse = lfm.get('coarse', 'other')
        conf       = lfm.get('conf', 0)

        # Only update if:
        # (a) currently 'other' / not_found and Last.fm gives a confident answer, OR
        # (b) Last.fm is significantly more confident than current (which has conf=1 by default)
        should_update = (
            (current == 'other' and new_coarse != 'other' and conf >= CONF_THRESHOLD)
        )

        if should_update:
            genre_labels[sid]['coarse']        = new_coarse
            genre_labels[sid]['lastfm_coarse'] = new_coarse
            genre_labels[sid]['lastfm_conf']   = conf
            genre_labels[sid]['lastfm_tags']   = lfm.get('track_tags', []) + lfm.get('artist_tags', [])
            if genre_labels[sid].get('source') == 'not_found':
                genre_labels[sid]['source'] = 'lastfm'
                improved += 1
            updated += 1

        else:
            # Always store Last.fm tags for reference even if we don't override
            genre_labels[sid]['lastfm_tags']   = lfm.get('track_tags', []) + lfm.get('artist_tags', [])
            genre_labels[sid]['lastfm_coarse'] = new_coarse
            genre_labels[sid]['lastfm_conf']   = conf

    with open(GENRE_PATH, 'w') as f:
        json.dump(genre_labels, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    new_coarse_dist = Counter(v.get('coarse') for v in genre_labels.values())
    log(f'\n=== Done ===')
    log(f'  Fetched tags for {len(tags_cache)} songs')
    log(f'  Updated {updated} coarse labels  ({improved} previously not_found)')
    log(f'\nNew genre distribution:')
    for genre, count in sorted(new_coarse_dist.items(), key=lambda x: -x[1]):
        log(f'  {genre:15s}: {count}')
    log(f'\nSaved updated labels to {GENRE_PATH}')


if __name__ == '__main__':
    main()
