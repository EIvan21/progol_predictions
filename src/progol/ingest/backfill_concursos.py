"""Walk quinielaposible.com to backfill historical Progol concursos.

We harvest (concurso_number, post_url) pairs from paginated category listings,
then scrape each post's 21-game slate and persist to progol_concursos +
progol_concurso_games. Fixture IDs are NOT resolved here — for old concursos
the API-Football date window approach in get_progol_ids.py would be wasteful.
A separate one-shot can backfill fixture_ids by joining team names against
the matches table once we have enough volume.

Usage:
    python -m src.progol.ingest.backfill_concursos                 # all available
    python -m src.progol.ingest.backfill_concursos --pages 5       # first 5 category pages
    python -m src.progol.ingest.backfill_concursos --min 2300      # stop at concurso 2300
"""
import argparse
import logging
import re
import time

import requests
from bs4 import BeautifulSoup

from src.progol import database
from src.progol.ingest.get_progol_ids import scrape_flexible_slate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SITEMAP_INDEX_URL = "https://quinielaposible.com/sitemap.xml"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# Slug variants we've seen across years (Yoast, multiple themes, redirects):
#   /progol-NNNN-quiniela-pronostico-y-fijos/
#   /pronostico-progol-NNNN/
#   /prediccion-progol-NNNN/
#   /progol-NNNN-resultados/      <-- post-game results, not slate, skip
SLUG_PATTERNS = [
    re.compile(r'/progol-(\d+)-quiniela'),
    re.compile(r'/pronostico-progol-(\d+)'),
    re.compile(r'/prediccion-progol-(\d+)'),
    re.compile(r'/progol-(\d+)-pronostico'),
]
RESULTS_PATTERN = re.compile(r'/progol-\d+-resultados')


def _list_post_sitemaps():
    res = requests.get(SITEMAP_INDEX_URL, headers=HEADERS, timeout=20)
    res.raise_for_status()
    return re.findall(r'<loc>(https://[^<]+/post-sitemap\d*\.xml)</loc>', res.text)


def harvest_post_urls():
    """Walk every post-sitemap and collect concurso_number -> URL.

    A concurso may have multiple posts (e.g., a pre-game pronostico AND a
    post-game resultados). Prefer pre-game pronostico URLs because they
    contain the 21-game slate; results posts are scoring summaries.
    Returns dict keyed by concurso_number."""
    seen = {}
    for sm in _list_post_sitemaps():
        try:
            res = requests.get(sm, headers=HEADERS, timeout=30)
        except Exception as exc:
            logger.warning(f"sitemap_fetch_failed {sm}: {exc}")
            continue
        urls = re.findall(r'<loc>([^<]+)</loc>', res.text)
        adds = 0
        for u in urls:
            if RESULTS_PATTERN.search(u):
                continue
            n = None
            for pat in SLUG_PATTERNS:
                m = pat.search(u)
                if m:
                    n = int(m.group(1))
                    break
            if n is None:
                continue
            if n not in seen:
                seen[n] = u
                adds += 1
        logger.info(f"{sm}: +{adds} (total {len(seen)})")
    return seen


def backfill_one(concurso_number, post_url):
    slate = scrape_flexible_slate(post_url)
    if not slate or len(slate) < 14:
        logger.warning(f"concurso {concurso_number}: slate too short ({len(slate)} games), skipping")
        return False
    games = [
        {'game_number': i + 1, 'home_name': h, 'away_name': v, 'fixture_id': None}
        for i, (h, v) in enumerate(slate[:21])
    ]
    database.upsert_concurso(
        concurso_number=concurso_number,
        week_start=None,
        source_url=post_url,
        games=games,
    )
    logger.info(f"concurso {concurso_number}: {len(games)} games persisted")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min', type=int, default=0, help='Stop at this concurso number (inclusive)')
    p.add_argument('--max', type=int, default=10**9, help='Skip concursos above this number')
    p.add_argument('--limit', type=int, default=None, help='Max concursos to backfill in this run')
    args = p.parse_args()

    database.init_db()
    posts = harvest_post_urls()
    targets = sorted(
        [(n, u) for n, u in posts.items() if args.min <= n <= args.max],
        reverse=True,
    )
    if args.limit:
        targets = targets[:args.limit]

    logger.info(f"Backfilling {len(targets)} concursos")
    ok = fail = 0
    for n, u in targets:
        try:
            if backfill_one(n, u):
                ok += 1
            else:
                fail += 1
        except Exception as exc:
            logger.warning(f"concurso {n} failed: {exc}")
            fail += 1
        time.sleep(0.5)
    logger.info(f"Backfill complete: ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
