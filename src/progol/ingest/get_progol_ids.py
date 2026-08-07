import requests
import re
import json
import os
import sys
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from thefuzz import process, fuzz
from dotenv import load_dotenv

from src.progol import config, database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
PROGOL_CAT_URL = "https://quinielaposible.com/category/progol/"
CACHE_FILE = config.PROGOL_IDS_PATH
EXPECTED_SLATE_SIZE = 21

NICKNAME_MAP = {
    "ÁGUILAS": "AMÉRICA", "AGUILAS": "AMERICA", "C. AZUL": "CRUZ AZUL",
    "PUMAS": "UNAM PUMAS", "CHIVAS": "GUADALAJARA",
    "ÁLAS": "ATLAS", "TIGRES": "TIGRES UANL", "RAYO VALLEC": "RAYO VALLECANO",
    "ATH. BILBAO": "ATHLETIC CLUB", "PARÍS F.C.": "PARIS FC", "LOS ÁNGELES": "LOS ANGELES FC",
    "ST. LOUIS": "ST. LOUIS CITY", "NIZA": "NICE",
    "COLONIA": "KOLN", "PSV": "PSV EINDHOVEN", "OFI CRETA": "OFI",
    "ARIS": "ARIS THESSALONIKIS", "NAPOLES": "NAPOLI", "AUGSBURGO": "AUGSBURG",
    "W. BREMEN": "WERDER BREMEN", "B. MUNICH": "BAYERN MUNICH", "INTER P.A.": "INTERNACIONAL",
    "VASCO DA GA": "VASCO DA GAMA", "SP. LISBOA": "SPORTING CP", "S. LAGUNA": "SANTOS LAGUNA",
    "MAN. UNITED": "MANCHESTER UNITED",
    "ATLANTA": "ATLANTA UNITED", "LA GALAXY": "LOS ANGELES GALAXY",
    "L.A. GALAXY": "LOS ANGELES GALAXY", "MINEIRO": "ATLETICO MINEIRO",
    "AT. MINEIRO": "ATLETICO MINEIRO", "ATLETICO-MG": "ATLETICO MINEIRO",
    # MLS short forms used on quinielaposible.com.
    "NY R. BULLS": "NEW YORK RED BULLS", "NY CITY": "NEW YORK CITY",
    "HOUSTON": "HOUSTON DYNAMO", "VANCOUVER": "VANCOUVER WHITECAPS",
    "MINNESOTA": "MINNESOTA UNITED", "SALT LAKE": "REAL SALT LAKE",
    "SAN DIEGO": "SAN DIEGO FC", "CHARLOTTE": "CHARLOTTE FC",
    "NEW ENGLAND": "NEW ENGLAND REVOLUTION", "TORONTO": "TORONTO FC",
    # Premier League — Man City missing in the existing list.
    "MAN. CITY": "MANCHESTER CITY",
    # Belgian Jupiler League: site uses Spanish/short forms.
    "BRUJAS": "BRUGGE", "UNION SG": "UNION SAINT-GILLOISE",
    # Scottish Premiership: site uses the short club name; API uses official.
    "HEARTS": "HEART OF MIDLOTHIAN",
    # Russian Premier League: site uses club-only; API uses "<club> Moscow" etc.
    "CSKA": "CSKA MOSCOW", "LOKOMOTIV": "LOKOMOTIV MOSCOW",
    "SPARTAK": "SPARTAK MOSCOW", "KRASNODAR": "FC KRASNODAR",
    "ZENIT": "ZENIT ST. PETERSBURG",
    # Chilean Primera: quinielaposible.com uses short-form university names.
    "U. CATOLICA": "UNIVERSIDAD CATOLICA", "U. DE CHILE": "UNIVERSIDAD DE CHILE",
    # J-League (Japan): site truncates to ~11 chars.
    "CEREZO OSAK": "CEREZO OSAKA", "GAMBA OSAK": "GAMBA OSAKA",
    "SHIMIZU": "SHIMIZU S-PULSE", "OKAYAMA": "FAGIANO OKAYAMA",
    "KASHIMA": "KASHIMA ANTLERS", "URAWA": "URAWA RED DIAMONDS",
    "YOKOHAMA FM": "YOKOHAMA F. MARINOS", "NAGOYA": "NAGOYA GRAMPUS",
    "SAPPORO": "HOKKAIDO CONSADOLE SAPPORO", "KAWASAKI": "KAWASAKI FRONTALE",
    "SANFRECCE": "SANFRECCE HIROSHIMA",
    # Women's football: site appends " F" suffix.
    "BARCELONA F": "BARCELONA FEMENI", "LYONNES F": "OLYMPIQUE LYONNAIS",
    "CHELSEA F": "CHELSEA WOMEN", "BAYERN F": "BAYERN MUNICH WOMEN",
    "WOLFSBURGO F": "WOLFSBURG WOMEN", "PSG F": "PARIS SAINT-GERMAIN WOMEN",
    # National teams: quinielaposible.com uses Spanish country names.
    "MEXICO": "MEXICO", "E.U.A.": "USA", "ESTADOS UNIDOS": "USA",
    "BRASIL": "BRAZIL", "NORUEGA": "NORWAY", "SUECIA": "SWEDEN",
    "JAPON": "JAPAN", "ISLANDIA": "ICELAND", "CHEQUIA": "CZECH REPUBLIC",
    "ALEMANIA": "GERMANY", "FRANCIA": "FRANCE", "INGLATERRA": "ENGLAND",
    "HOLANDA": "NETHERLANDS", "PAISES BAJOS": "NETHERLANDS",
    "ESCOCIA": "SCOTLAND", "GALES": "WALES", "IRLANDA": "IRELAND",
    "DINAMARCA": "DENMARK", "SUIZA": "SWITZERLAND", "BELGICA": "BELGIUM",
    "AUSTRIA": "AUSTRIA", "CROACIA": "CROATIA", "SERBIA": "SERBIA",
    "TURQUIA": "TURKEY", "GRECIA": "GREECE", "RUMANIA": "ROMANIA",
    "HUNGRIA": "HUNGARY", "POLONIA": "POLAND", "COREA SUR": "SOUTH KOREA",
    "COREA DEL SUR": "SOUTH KOREA", "CHINA": "CHINA",
    "COLOMBIA": "COLOMBIA", "ARGENTINA": "ARGENTINA", "CHILE": "CHILE",
    "PERU": "PERU", "URUGUAY": "URUGUAY", "PARAGUAY": "PARAGUAY",
    "VENEZUELA": "VENEZUELA", "BOLIVIA": "BOLIVIA", "ECUADOR": "ECUADOR",
    "PANAMA": "PANAMA", "COSTA RICA": "COSTA RICA",
    "A. SAUDITA": "SAUDI ARABIA", "ARABIA SAUDITA": "SAUDI ARABIA",
    "MARRUECOS": "MOROCCO", "TUNEZ": "TUNISIA", "ARGELIA": "ALGERIA",
    "SENEGAL": "SENEGAL", "CAMERUN": "CAMEROON", "NIGERIA": "NIGERIA",
    "GHANA": "GHANA", "COSTA MARFIL": "IVORY COAST",
    "AUSTRALIA": "AUSTRALIA", "NUEVA ZELANDA": "NEW ZEALAND",
    "CANADA": "CANADA", "JAMAICA": "JAMAICA", "HONDURAS": "HONDURAS",
    "EL SALVADOR": "EL SALVADOR", "GUATEMALA": "GUATEMALA",
    "KOSOVO": "KOSOVO", "ESLOVENIA": "SLOVENIA", "ESLOVAQUIA": "SLOVAKIA",
    "PORTUGAL": "PORTUGAL", "ITALIA": "ITALY", "ESPAÑA": "SPAIN",
    # 2026 World Cup slate names absent from the country list above; API
    # uses these exact long forms, so map to them for the fuzzy match.
    "EGIPTO": "EGYPT", "CABO VERDE": "CAPE VERDE ISLANDS",
    "BOSNIA": "BOSNIA & HERZEGOVINA", "REP. CONGO": "CONGO DR",
    # Brazil Serie B: site truncates club names.
    "AVAI": "AVAI FC", "CRICIUMA": "CRICIUMA", "ATL. GO": "ATLETICO GOIANIENSE",
    "GOIAS": "GOIAS", "VITORIA BA": "VITORIA",
    # Swedish Allsvenskan: site truncates heavily.
    "DEGERFORS": "DEGERFORS IF", "BROMMAPOJ": "BROMMAPOJKARNA",
}

def clean_name(name):
    name = name.upper().strip()
    for nickname, official in NICKNAME_MAP.items():
        if nickname in name: return official
    removals = ["FC", "CF", "CD", "UD", "CLUB", "DEPORTIVO", "REAL", "ATLETICO", "S.C."]
    for r in removals: name = name.replace(r, "").strip()
    return name

def get_latest_progol_post_url():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(PROGOL_CAT_URL, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        candidates = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            m = re.search(r'/progol-(\d+)[\w\-]*/?$', href)
            if m and not any(ext in href for ext in ['.png', '.jpg', '.jpeg', '.svg']):
                candidates.append((int(m.group(1)), href))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
    except: pass
    return PROGOL_CAT_URL

def scrape_flexible_slate(url):
    logging.info(f"Scraping official slate from: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    matches = []
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Slate table on quinielaposible.com has 5 cols: [prob_L, LOCAL, prob_E, VISIT, prob_V]
        # 14 main rows + "Revancha" divider + 7 revancha rows = 21 unique matches.
        for table in soup.find_all('table'):
            local_matches = []
            for row in table.find_all('tr'):
                cols = [c.get_text(strip=True) for c in row.find_all('td')]
                if len(cols) == 5:
                    home, away = cols[1], cols[3]
                    if home and away and not home.replace('.', '').isdigit():
                        local_matches.append((home, away))
            if len(local_matches) >= 14:
                matches = local_matches
                break

        return matches[:21]
    except Exception as exc:
        logging.error(f"scrape_failed: {exc}")
        return []


def get_upcoming_api_fixtures(days_back=3, days_forward=12):
    """Fetch fixtures in a date window (played + unplayed). The previous version
    used `next=50` which silently dropped fixtures that had already kicked off,
    leaving the slate resolver with <21 matches by Saturday afternoon.

    days_forward=12 (was 5→8→12): Progol slates can span up to ~2 weeks
    (friendlies, cup finals). 12 days from the Wednesday run covers through
    the following Sunday+Monday comfortably.
    days_back=3 (was 2): some Friday fixtures are already FT by the
    Wednesday run — widen to catch them for resolution."""
    headers = {"x-apisports-key": API_KEY}
    today = datetime.now().date()
    date_from = (today - timedelta(days=days_back)).isoformat()
    date_to = (today + timedelta(days=days_forward)).isoformat()
    leagues = [
        # Domestic leagues
        262,  # Liga MX
        39,   # Premier League
        140,  # La Liga
        135,  # Serie A
        78,   # Bundesliga
        61,   # Ligue 1
        253,  # MLS
        71,   # Brazil Serie A
        128,  # Argentina Primera
        94,   # Portugal Primeira Liga
        88,   # Eredivisie
        144,  # Belgium Jupiler
        40,   # Championship
        141,  # La Liga 2
        197,  # Greek Super League
        79,   # Bundesliga 2
        263,  # Liga MX Expansion
        179,  # Scottish Premiership
        235,  # Russian Premier League
        265,  # Chilean Primera División
        98,   # J1 League (Japan)
        99,   # J2 League (Japan)
        # Cup competitions
        13,   # Copa Libertadores
        2,    # UEFA Champions League
        3,    # UEFA Europa League
        11,   # Copa Sudamericana
        45,   # FA Cup
        48,   # EFL Cup
        66,   # Coupe de France
        81,   # DFB-Pokal
        137,  # Coppa Italia
        143,  # Copa del Rey
        # Women's competitions
        750,  # Women's Champions League (UWCL)
        # CONCACAF / International
        16,   # CONCACAF Champions Cup (Concachampions)
        10,   # Friendlies (international)
        772,  # Leagues Cup (Liga MX vs MLS)
        667,  # Club Friendlies (preseason: Juventus, Chelsea, etc.)
        1,    # FIFA World Cup — national-team slates (e.g. concurso 2339)
        # Additional domestic leagues that appear in Progol slates
        103,  # Norway Eliteserien
        268,  # Uruguay Primera División
        242,  # Ecuador Liga Pro
        72,   # Brazil Serie B
        113,  # Swedish Allsvenskan
        114,  # Swedish Superettan
    ]
    all_f = []
    for lid in leagues:
        for sn in [2025, 2026]:
            try:
                url = f"{BASE_URL}/fixtures?league={lid}&season={sn}&from={date_from}&to={date_to}"
                res = requests.get(url, headers=headers, timeout=20).json()
                all_f.extend(res.get('response', []))
            except Exception as exc:
                logging.warning(f"fixture_fetch_failed league={lid} season={sn}: {exc}")
                continue
    logging.info(f"Fetched {len(all_f)} candidate fixtures in window {date_from} -> {date_to}")
    return all_f

def resolve_matches(scraped, api_data, threshold=70):
    """Resolve scraped (home, away) pairs to API fixture IDs.

    Returns a list of dicts: {game_number, home_name, away_name, fixture_id, inverted}.
    home_name/away_name are the ORIGINAL scraped names (preserves Progol ordering)
    so the L/E/V labels stay consistent when we persist to progol_concurso_games.
    """
    fwd_map = {f"{clean_name(f['teams']['home']['name'])} vs {clean_name(f['teams']['away']['name'])}": f['fixture']['id'] for f in api_data}
    rev_map = {f"{clean_name(f['teams']['away']['name'])} vs {clean_name(f['teams']['home']['name'])}": f['fixture']['id'] for f in api_data}

    fwd_keys = list(fwd_map.keys())
    rev_keys = list(rev_map.keys())
    games = []
    print(f"\nRESOLVING 21-MATCH SLATE (threshold={threshold})...")
    for i, (h, v) in enumerate(scraped):
        query = f"{clean_name(h)} vs {clean_name(v)}"
        fwd_match, fwd_score = process.extractOne(query, fwd_keys, scorer=fuzz.token_sort_ratio) if fwd_keys else (None, 0)
        rev_match, rev_score = process.extractOne(query, rev_keys, scorer=fuzz.token_sort_ratio) if rev_keys else (None, 0)

        fid = None
        inverted = False
        best = max(fwd_score, rev_score)
        if fwd_score >= threshold and fwd_score >= rev_score:
            fid, inverted, score = fwd_map[fwd_match], False, fwd_score
            tag = " [LOW-CONFIDENCE]" if score < 75 else ""
            print(f"Game {i+1:2}: {h:15} vs {v:15} -> ID {fid} (score {score}){tag}")
            if score < 75:
                logging.warning(f"low_confidence_match game={i+1} {h} vs {v} score={score} matched={fwd_match}")
        elif rev_score >= threshold:
            fid, inverted, score = rev_map[rev_match], True, rev_score
            tag = " [LOW-CONFIDENCE]" if score < 75 else ""
            print(f"Game {i+1:2}: {h:15} vs {v:15} -> ID {fid} (score {score})  [INVERTED]{tag}")
            if score < 75:
                logging.warning(f"low_confidence_match game={i+1} {h} vs {v} score={score} matched={rev_match} inverted")
        else:
            print(f"Game {i+1:2}: {h:15} vs {v:15} -> FAILED (best score: {best})")

        games.append({
            'game_number': i + 1,
            'home_name': h,
            'away_name': v,
            'fixture_id': fid,
            'inverted': inverted,
        })
    return games


def parse_concurso_number(url):
    """Extract Progol concurso number from a quinielaposible.com URL like
    https://quinielaposible.com/progol-2331-..."""
    if not url:
        return None
    m = re.search(r'/progol-(\d+)', url)
    return int(m.group(1)) if m else None

if __name__ == "__main__":
    post_url = get_latest_progol_post_url()
    slate = scrape_flexible_slate(post_url)
    if not slate:
        logging.error("Slate not found.")
        sys.exit(1)

    api_data = get_upcoming_api_fixtures()
    games = resolve_matches(slate, api_data)
    resolved = [(g['game_number'], g['fixture_id']) for g in games if g['fixture_id'] is not None]
    resolved_ids = [fid for _, fid in resolved]
    resolved_game_numbers = [gn for gn, _ in resolved]
    concurso_number = parse_concurso_number(post_url)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # `game_numbers` mirrors `match_ids` 1:1 and carries the original
    # 1..21 Progol slot for each fixture. predict.py needs it to write
    # predicted_probs to the right concurso row when some slate entries
    # fail to resolve — otherwise the i-th resolved prediction lands at
    # DB game_number=i, misaligning labels for any unresolved game before it.
    payload = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "source_url": post_url,
        "concurso_number": concurso_number,
        "expected": EXPECTED_SLATE_SIZE,
        "resolved": len(resolved_ids),
        "match_ids": resolved_ids,
        "game_numbers": resolved_game_numbers,
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(payload, f, indent=2)

    if concurso_number:
        try:
            database.init_db()
            database.upsert_concurso(
                concurso_number=concurso_number,
                week_start=datetime.now().strftime("%Y-%m-%d"),
                source_url=post_url,
                games=games,
            )
            logging.info(f"Persisted concurso {concurso_number} ({len(games)} games) to DB")
        except Exception as exc:
            logging.warning(f"upsert_concurso failed: {exc}")

    logging.info(f"Resolved {len(resolved_ids)}/{EXPECTED_SLATE_SIZE} matches -> {CACHE_FILE}")
    if len(resolved_ids) != EXPECTED_SLATE_SIZE:
        logging.error(
            f"Slate incomplete: {len(resolved_ids)}/{EXPECTED_SLATE_SIZE} resolved. "
            f"Check NICKNAME_MAP and the date window in get_upcoming_api_fixtures."
        )
        sys.exit(2)
