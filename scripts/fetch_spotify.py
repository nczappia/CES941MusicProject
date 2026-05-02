"""
Fetch Spotify metadata for all songs in the dataset.

Uses Client Credentials flow (no user login needed).
Searches by title + artist, picks the top match, then fetches:
  - Artist genres (fine-grained Spotify taxonomy)
  - Track popularity (0-100)
  - Release year
  - Audio features (key, mode, tempo, energy, etc.) — requires extended access;
    skipped gracefully with a 403 if not yet approved.

Output: data/spotify_features.json
    {song_id: {title, artist, spotify_id, spotify_genres, popularity,
               release_year, audio_features}}

Resumable — skips songs already in the output file.

Run from project root:
    source venv/bin/activate && python scripts/fetch_spotify.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

GENRE_PATH  = 'data/genre_labels.json'
OUTPUT_PATH = 'data/spotify_features.json'
LOG_PATH    = 'results/fetch_spotify.log'

CLIENT_ID     = os.environ.get('SPOTIPY_CLIENT_ID',     '24b6166fce6d4e8caa354e82dbf7648b')
CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET', '17323f87807a48d99d036f118d5c9e77')

AUDIO_FEATURE_KEYS = [
    'key', 'mode', 'tempo', 'energy', 'danceability',
    'valence', 'acousticness', 'instrumentalness',
    'liveness', 'speechiness', 'loudness', 'time_signature',
]


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(msg + '\n')


def search_track(sp, title, artist):
    """Return the best-matching track object or None."""
    for query in [f'track:{title} artist:{artist}', f'track:{title}']:
        try:
            r = sp.search(q=query, type='track', limit=1)
            items = r['tracks']['items']
            if items:
                return items[0]
        except Exception as e:
            log(f'  search error: {e}')
    return None


def fetch_artist_genres(sp, artist_id):
    """Return Spotify genre tags for an artist."""
    try:
        artist = sp.artist(artist_id)
        return artist.get('genres', [])
    except Exception:
        return []


def fetch_audio_features(sp, track_id):
    """Return audio features dict or None. Fails gracefully if 403 (no extended access)."""
    try:
        feats = sp.audio_features([track_id])[0]
        if feats:
            return {k: feats[k] for k in AUDIO_FEATURE_KEYS if k in feats}
    except Exception as e:
        if '403' in str(e):
            return None   # extended access not yet granted — silent skip
        log(f'  audio_features error: {e}')
    return None


def main():
    os.makedirs('results', exist_ok=True)

    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    ))

    # Quick check: does audio_features work yet?
    test_id = '1h2xVEoJORqrg71HocgqXd'  # Superstition
    audio_access = fetch_audio_features(sp, test_id) is not None
    log(f'Spotify client OK | audio_features access: {"YES" if audio_access else "NO (extended access pending)"}')

    with open(GENRE_PATH) as f:
        genre_data = json.load(f)

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
        log(f'Resuming — {len(results)} songs already fetched')
    else:
        results = {}

    # Re-fetch any entries that are missing audio_features if access was just granted
    if audio_access:
        needs_audio = [sid for sid, v in results.items()
                       if v.get('spotify_id') and v.get('audio_features') is None]
        if needs_audio:
            log(f'Back-filling audio features for {len(needs_audio)} previously fetched songs...')
            for sid in needs_audio:
                feats = fetch_audio_features(sp, results[sid]['spotify_id'])
                results[sid]['audio_features'] = feats
                if feats:
                    log(f'  backfill OK — {results[sid]["artist"]} — {results[sid]["title"]}')
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(results, f, indent=2)

    songs = [(sid, info) for sid, info in genre_data.items() if sid not in results]
    log(f'Fetching {len(songs)} remaining songs...\n')

    for i, (sid, info) in enumerate(songs):
        title  = info.get('title', '')
        artist = info.get('artist', '')

        if not title or not artist:
            results[sid] = {
                'title': title, 'artist': artist,
                'spotify_id': None, 'spotify_genres': [],
                'popularity': None, 'release_year': None, 'audio_features': None,
            }
            continue

        track = search_track(sp, title, artist)

        if track is None:
            results[sid] = {
                'title': title, 'artist': artist,
                'spotify_id': None, 'spotify_genres': [],
                'popularity': None, 'release_year': None, 'audio_features': None,
            }
            log(f'[{i+1}/{len(songs)}] MISS — {artist} — {title}')
        else:
            track_id   = track['id']
            popularity = track.get('popularity')
            release    = track.get('album', {}).get('release_date', '')
            release_year = int(release[:4]) if release else None
            artist_id  = track['artists'][0]['id']

            genres   = fetch_artist_genres(sp, artist_id)
            features = fetch_audio_features(sp, track_id) if audio_access else None

            results[sid] = {
                'title':          title,
                'artist':         artist,
                'spotify_id':     track_id,
                'spotify_genres': genres,
                'popularity':     popularity,
                'release_year':   release_year,
                'audio_features': features,
            }
            feat_status = 'audio+meta' if features else 'meta only'
            log(f'[{i+1}/{len(songs)}] OK ({feat_status}) — {artist} — {title} | genres: {genres[:3]}')

        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)

        time.sleep(0.15)  # conservative rate limiting

    matched_meta  = sum(1 for v in results.values() if v.get('spotify_id'))
    matched_audio = sum(1 for v in results.values() if v.get('audio_features'))
    log(f'\nDone. {matched_meta}/{len(results)} matched | {matched_audio} with audio features.')
    log(f'Saved to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
