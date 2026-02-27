import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('FOOTBALL_API_KEY')
BASE_URL = "https://v3.football.api-sports.io"
headers = {"x-apisports-key": API_KEY}

def test_api():
    print("--- API DIAGNOSTICS ---")
    
    # 1. Test Liga MX (262) for 2025/2026
    print("\nTesting Liga MX (262) Season 2025...")
    url_2025 = f"{BASE_URL}/fixtures?league=262&season=2025&next=5"
    res_2025 = requests.get(url_2025, headers=headers).json()
    print(f"Results for 2025: {len(res_2025.get('response', []))}")
    
    print("\nTesting Liga MX (262) Season 2026...")
    url_2026 = f"{BASE_URL}/fixtures?league=262&season=2026&next=5"
    res_2026 = requests.get(url_2026, headers=headers).json()
    print(f"Results for 2026: {len(res_2026.get('response', []))}")

    # 2. Test Today's Date directly for Liga MX
    today = "2026-02-27"
    print(f"\nTesting Date {today} for Liga MX...")
    url_date = f"{BASE_URL}/fixtures?league=262&date={today}"
    res_date = requests.get(url_date, headers=headers).json()
    print(f"Results for {today}: {len(res_date.get('response', []))}")

    # 3. Check if we can get a list of active leagues for 2026
    print("\nChecking active leagues for 2026...")
    url_leagues = f"{BASE_URL}/leagues?season=2026"
    res_leagues = requests.get(url_leagues, headers=headers).json()
    leagues = [f"{l['league']['name']} ({l['league']['id']})" for l in res_leagues.get('response', [])[:10]]
    print(f"Top 10 Active Leagues in 2026: {leagues}")

if __name__ == "__main__":
    test_api()
