"""
Fetch genre tags for McGill Billboard songs via MusicBrainz API.

No API key required. Rate-limited to 1 req/sec per MusicBrainz policy.
Saves results progressively so the script is safe to interrupt and resume.

Run from project root:
    python scripts/fetch_genres.py

Output:
    data/genre_labels.json   — {song_id: {title, artist, genres: [...], source}}
"""

import sys, os, json, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from pathlib import Path
from src.parse import load_all_songs

DATA_DIR   = 'data/McGill-Billboard'
OUTPUT     = 'data/genre_labels.json'
MB_BASE    = 'https://musicbrainz.org/ws/2'
USER_AGENT = 'CES941MusicProject/1.0 (nick.zappia@msu.edu)'

HEADERS = {
    'User-Agent': USER_AGENT,
    'Accept': 'application/json',
}


def clean(s: str) -> str:
    """Strip parentheticals and featured artists for cleaner matching."""
    s = re.sub(r'\s*\(.*?\)', '', s)
    s = re.sub(r'\s*feat\..*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*ft\..*',   '', s, flags=re.IGNORECASE)
    return s.strip()


def mb_search_recording(title: str, artist: str) -> list:
    """
    Search MusicBrainz for a recording and return a list of genre/tag strings.
    Returns empty list on failure.
    """
    query = f'recording:"{clean(title)}" AND artist:"{clean(artist)}"'
    params = {
        'query': query,
        'fmt':   'json',
        'limit': 3,
    }
    try:
        r = requests.get(f'{MB_BASE}/recording', params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f'    [error] {e}')
        return []

    recordings = data.get('recordings', [])
    if not recordings:
        return []

    # Collect tags from all top results, ranked by score
    tag_counts: dict = {}
    for rec in recordings[:3]:
        if int(rec.get('score', 0)) < 60:
            continue
        for tag in rec.get('tags', []):
            name = tag['name'].lower().strip()
            count = tag.get('count', 1)
            tag_counts[name] = tag_counts.get(name, 0) + count

    if not tag_counts:
        return []

    # Return tags sorted by count descending
    return [t for t, _ in sorted(tag_counts.items(), key=lambda x: -x[1])]


def mb_search_artist_tags(artist: str) -> list:
    """Fallback: get genre tags from the artist entity."""
    params = {
        'query': f'artist:"{clean(artist)}"',
        'fmt':   'json',
        'limit': 1,
    }
    try:
        r = requests.get(f'{MB_BASE}/artist', params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    artists = data.get('artists', [])
    if not artists or int(artists[0].get('score', 0)) < 70:
        return []

    return [t['name'].lower() for t in artists[0].get('tags', [])]


# High-level genre buckets for normalization
GENRE_MAP = {
    'rock':        ['rock', 'classic rock', 'hard rock', 'soft rock', 'folk rock',
                    'country rock', 'psychedelic rock', 'punk', 'punk rock', 'alternative',
                    'alternative rock', 'indie rock', 'grunge', 'new wave'],
    'pop':         ['pop', 'pop rock', 'dance pop', 'teen pop', 'synth-pop', 'bubblegum'],
    'soul_r&b':    ['soul', 'r&b', 'rhythm and blues', 'funk', 'motown', 'neo soul',
                    'gospel', 'urban'],
    'country':     ['country', 'country pop', 'bluegrass', 'americana', 'country rock'],
    'jazz':        ['jazz', 'jazz pop', 'smooth jazz', 'bossa nova', 'swing', 'bebop'],
    'blues':       ['blues', 'blues rock', 'electric blues', 'chicago blues'],
    'hip_hop':     ['hip hop', 'hip-hop', 'rap', 'hip hop soul'],
    'disco_dance': ['disco', 'dance', 'electronic', 'house', 'new wave'],
    'folk':        ['folk', 'folk rock', 'singer-songwriter', 'acoustic'],
}


def coarse_genre(tags: list) -> str:
    """Map a list of fine-grained tags to a coarse genre bucket."""
    for tag in tags:
        for coarse, fine_list in GENRE_MAP.items():
            if any(f in tag for f in fine_list) or tag in fine_list:
                return coarse
    return 'other'


def main():
    # ── Load existing results (resume support) ────────────────────────────
    if Path(OUTPUT).exists():
        with open(OUTPUT) as f:
            results = json.load(f)
        print(f'Resuming — {len(results)} songs already fetched')
    else:
        results = {}

    # ── Load songs ────────────────────────────────────────────────────────
    print('Loading songs...')
    songs = load_all_songs(DATA_DIR)
    print(f'  {len(songs)} songs to process')

    to_fetch = [s for s in songs if s['song_id'] not in results]
    print(f'  {len(to_fetch)} remaining')

    # ── Fetch ─────────────────────────────────────────────────────────────
    for i, song in enumerate(to_fetch):
        sid    = song['song_id']
        title  = song.get('title', '')
        artist = song.get('artist', '')

        if not title or not artist:
            results[sid] = {'title': title, 'artist': artist, 'genres': [], 'coarse': 'unknown', 'source': 'no_metadata'}
            continue

        print(f'[{i+1}/{len(to_fetch)}] {artist} — {title}', end=' ... ', flush=True)

        # Artist-level tags are much more consistently populated than recording-level
        tags = mb_search_artist_tags(artist)
        source = 'artist'

        if not tags:
            # Fallback to recording-level tags
            tags = mb_search_recording(title, artist)
            source = 'recording_fallback' if tags else 'not_found'

        coarse = coarse_genre(tags)
        results[sid] = {
            'title':   title,
            'artist':  artist,
            'genres':  tags[:10],    # store top 10 fine-grained tags
            'coarse':  coarse,
            'source':  source,
        }
        print(f'{coarse} ({", ".join(tags[:3]) if tags else "—"})')

        # Save after every song so we can resume
        with open(OUTPUT, 'w') as f:
            json.dump(results, f, indent=2)

        # MusicBrainz rate limit: max 1 req/sec (we make up to 2 per song)
        time.sleep(1.1)

    # ── Summary ───────────────────────────────────────────────────────────
    print('\n=== Genre distribution ===')
    from collections import Counter
    coarse_counts = Counter(v['coarse'] for v in results.values())
    for genre, count in sorted(coarse_counts.items(), key=lambda x: -x[1]):
        print(f'  {genre:<15s}: {count}')

    found = sum(1 for v in results.values() if v['source'] != 'not_found' and v['source'] != 'no_metadata')
    print(f'\nMatch rate: {found}/{len(results)} = {found/max(len(results),1)*100:.1f}%')
    print(f'Saved to {OUTPUT}')


if __name__ == '__main__':
    main()
