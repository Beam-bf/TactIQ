"""Download a few open StatsBomb matches and produce public/matches.json.

Usage: python analytics/import_statsbomb.py --limit 3
Requires internet access at execution time. Standard library only.
"""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen, Request

BASE = "https://raw.githubusercontent.com/hudl/open-data/master/data"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "public" / "matches.json"


def fetch(path):
    req = Request(f"{BASE}/{path}", headers={"User-Agent": "TactiQ-AI-educational-prototype"})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def convert(match):
    events = fetch(f"events/{match['match_id']}.json")
    passes = []
    for e in events:
        if e.get("type", {}).get("name") != "Pass":
            continue
        start = e.get("location")
        ending = e.get("pass", {}).get("end_location")
        if not start or not ending or len(start) < 2 or len(ending) < 2:
            continue
        p = e["pass"]
        passes.append({
            "team": e.get("team", {}).get("name", "Unknown"),
            "player": e.get("player", {}).get("name", "Unknown"),
            "recipient": p.get("recipient", {}).get("name", "Unknown"),
            "x": float(start[0]), "y": float(start[1]),
            "endX": float(ending[0]), "endY": float(ending[1]),
            "completed": "outcome" not in p,
            "minute": int(e.get("minute", 0))
        })
    return {
        "id": str(match["match_id"]),
        "label": f"{match['home_team']['home_team_name']} vs {match['away_team']['away_team_name']}",
        "competition": "World Cup 2018 (StatsBomb Open Data)",
        "date": match.get("match_date", "Unknown"),
        "home": match["home_team"]["home_team_name"],
        "away": match["away_team"]["away_team_name"],
        "passes": passes,
        "source": "statsbomb"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.limit <= 12:
        parser.error("--limit must be between 1 and 12")
    print("Loading World Cup 2018 match list...")
    matches = fetch("matches/43/3.json")
    converted = []
    for match in matches[:args.limit]:
        print(f"Importing match {match['match_id']}...")
        converted.append(convert(match))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(converted, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {len(converted)} matches to {OUT}")
    print("Restart/refresh the frontend to view real match passes.")


if __name__ == "__main__":
    main()
