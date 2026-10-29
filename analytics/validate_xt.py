"""Validate whether higher observed possession xT is associated with shots.

Run from project root:
    python -m pip install numpy pandas scikit-learn
    python analytics/validate_xt.py --limit 50

Reads the project's existing 8 x 12 xT grid; downloads StatsBomb 2018 WC
open-event files. Does NOT change the trained model or frontend code.
"""
import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
GRID_FILE = ROOT / 'frontend' / 'src' / 'xt-grid.json'
OUT_FILE = ROOT / 'frontend' / 'public' / 'xt-validation.json'
BASE = 'https://raw.githubusercontent.com/hudl/open-data/master/data'


def fetch(path):
    request = Request(f'{BASE}/{path}', headers={'User-Agent': 'TactiQ-AI-academic-analytics/1.0'})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def xt_value(grid, xy):
    if not isinstance(xy, list) or len(xy) < 2:
        return None
    try:
        x, y = float(xy[0]), float(xy[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 120 and 0 <= y <= 80):
        return None
    col = min(11, int(x / 10))
    row = min(7, int(y / 10))
    return float(grid[row][col])


def possession_records(match_id, grid):
    events = fetch(f'events/{match_id}.json')
    records = {}
    for event in events:
        pid = event.get('possession')
        team = (event.get('possession_team') or {}).get('name')
        if pid is None or not team:
            continue
        key = (str(match_id), str(pid), team)
        r = records.setdefault(key, {
            'match_id': str(match_id), 'possession': str(pid), 'team': team,
            'xt_positive': 0., 'xt_signed': 0., 'completed_passes': 0,
            'had_shot': 0, 'shot_xg_sum': 0.,
        })
        event_type = (event.get('type') or {}).get('name')
        event_team = (event.get('team') or {}).get('name')
        # Credit passes/shots only to the team recorded in possession.
        if event_team != team:
            continue
        if event_type == 'Pass':
            p = event.get('pass') or {}
            # No outcome usually indicates completion in StatsBomb event files.
            if p.get('outcome') is not None:
                continue
            start = xt_value(grid, event.get('location'))
            end = xt_value(grid, p.get('end_location'))
            if start is None or end is None:
                continue
            delta = end - start
            r['xt_signed'] += delta
            r['xt_positive'] += max(0., delta)
            r['completed_passes'] += 1
        elif event_type == 'Shot':
            r['had_shot'] = 1
            try:
                value = float((event.get('shot') or {}).get('statsbomb_xg', 0))
                if math.isfinite(value) and value >= 0:
                    r['shot_xg_sum'] += value
            except (ValueError, TypeError):
                pass
    return list(records.values())


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 6) if math.isfinite(float(obj)) else None
    return obj


def analyze(df):
    # Quantiles of ranked values keep ties, especially zero-xT possessions, manageable.
    df = df.copy()
    df['bucket'] = pd.qcut(df['xt_positive'].rank(method='first'), 5, labels=False) + 1
    summary = []
    for bucket, part in df.groupby('bucket', sort=True):
        summary.append({
            'quintile': int(bucket),
            'possessions': len(part),
            'shot_rate': float(part['had_shot'].mean()),
            'mean_positive_xt': float(part['xt_positive'].mean()),
            'mean_signed_xt': float(part['xt_signed'].mean()),
            'mean_shot_xg': float(part['shot_xg_sum'].mean()),
        })
    labels = df['had_shot'].to_numpy()
    return {
        'dataset': 'StatsBomb Open Data — 2018 World Cup',
        'method': 'Sum of positive completed-pass xT gains per possession; same-possession shot association',
        'matches': int(df['match_id'].nunique()),
        'possessions': int(len(df)),
        'possessions_with_shot': int(df['had_shot'].sum()),
        'overall_shot_rate': float(df['had_shot'].mean()),
        'shot_discrimination_auc': float(roc_auc_score(labels, df['xt_positive'])) if len(np.unique(labels)) == 2 else None,
        'quintiles': summary,
        'limitations': [
            'Descriptive association, not proof that passing xT causes shots or goals.',
            'Positive xT counts only successful passes and excludes carries, dribbles, and turnovers.',
            'Shot and passing events are aggregated across the same possession; this is not a prospective prediction.',
            'Rank-based quintiles may split tied zero-xT possessions between groups.',
            'StatsBomb 120x80 attack-relative event coordinates and the supplied xT grid are assumed compatible.',
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50, help='Maximum World Cup matches to download')
    args = parser.parse_args()
    if args.limit < 5:
        parser.error('Use at least 5 matches for meaningful group comparisons.')
    if not GRID_FILE.exists():
        raise FileNotFoundError(f'Missing grid: {GRID_FILE}')
    grid = np.asarray(json.loads(GRID_FILE.read_text(encoding='utf-8')), dtype=float)
    if grid.shape != (8, 12) or not np.isfinite(grid).all():
        raise ValueError('Expected a finite 8-row x 12-column xT grid.')
    matches = fetch('matches/43/3.json')
    rows = []
    for match in matches[:args.limit]:
        mid = str(match['match_id'])
        try:
            match_rows = possession_records(mid, grid)
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            print(f'WARNING: skipped {mid}: {exc}')
            continue
        rows.extend(match_rows)
        print(f'Loaded {mid}: {len(match_rows)} possessions')
    df = pd.DataFrame(rows)
    if df.empty or df['match_id'].nunique() < 5 or len(df) < 50:
        raise RuntimeError('Not enough usable possessions. Check connection and available matches.')
    result = analyze(df)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(to_jsonable(result), indent=2), encoding='utf-8')
    print('\nXT VS. SHOTS VALIDATION')
    print(json.dumps(to_jsonable(result), indent=2))
    print(f'\nExported {OUT_FILE}')
    print('NOTE: Compare quintile shot rates; a positive association is not proof of causality.')


if __name__ == '__main__':
    main()
