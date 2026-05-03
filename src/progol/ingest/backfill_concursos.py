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

CATEGORY_URL = "https://quinielaposible.com/category/progol/"
HEADERS = {'User-Agent': 'Mozilla/5.0'}


def harvest_post_urls(max_pages=20):
    """Walk paginated category pages and collect concurso_number -> URL.
    Stops early if a page yields nothing new."""
    seen = {}
    for page in range(1, max_pages + 1):
        url = CATEGORY_URL if page == 1 else f"{CATEGORY_URL}page/{page}/"
        try:
            res = requests.get(url, headers=HEADERS, timeout=20)
        except Exception as exc:
            logger.warning(f"page_fetch_failed page={page}: {exc}")
            break
        if res.status_code != 200:
            logger.info(f"page {page} -> HTTP {res.status_code}, stopping")
            break

        soup = BeautifulSoup(res.text, 'html.parser')
        new_on_page = 0
        for a in soup.find_all('a', href=True):
            href = a['href']
            m = re.search(r'/progol-(\d+)[\w\-]*/?$', href)
            if not m or any(ext in href for ext in ['.png', '.jpg', '.jpeg', '.svg']):
                continue
            n = int(m.group(1))
            if n not in seen:
                seen[n] = href
                new_on_page += 1
        logger.info(f"page {page}: +{new_on_page} (total {len(seen)})")
        if new_on_page == 0:
            break
        time.sleep(0.5)
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
    p.add_argument('--pages', type=int, default=20, help='Max category pages to walk')
    p.add_argument('--min', type=int, default=0, help='Stop at this concurso number (inclusive)')
    p.add_argument('--max', type=int, default=10**9, help='Skip concursos above this number')
    p.add_argument('--limit', type=int, default=None, help='Max concursos to backfill in this run')
    args = p.parse_args()

    database.init_db()
    posts = harvest_post_urls(max_pages=args.pages)
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
