"""Train a pass-completion classifier on StatsBomb World Cup 2018 open event data.

Requires: pip install numpy scikit-learn
Run from project root: python analytics/train_pass_model.py --limit 50
Downloads public events; outputs frontend/public/pass-model.json.
"""
import argparse
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = 'https://raw.githubusercontent.com/hudl/open-data/master/data'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'frontend' / 'public' / 'pass-model.json'
FEATURES = ['x', 'y', 'dx', 'dy', 'distance', 'minute']

def fetch(path):
    request = Request(f'{BASE}/{path}', headers={'User-Agent': 'TactiQ-AI-academic-analytics/1.0'})
    with urlopen(request, timeout=45) as response:
        return json.load(response)

def parse_match(match_id):
    events = fetch(f'events/{match_id}.json')
    X, y = [], []
    for event in events:
        if event.get('type', {}).get('name') != 'Pass':
            continue
        p = event.get('pass') or {}
        start, end = event.get('location'), p.get('end_location')
        if not start or not end or len(start) < 2 or len(end) < 2:
            continue
        x, side, ex, ey = map(float, (start[0], start[1], end[0], end[1]))
        if not all(math.isfinite(v) for v in (x, side, ex, ey)):
            continue
        if not (0 <= x <= 120 and 0 <= ex <= 120 and 0 <= side <= 80 and 0 <= ey <= 80):
            continue
        dx, dy = ex - x, ey - side
        minute = float(event.get('minute', 0))
        X.append([x, side, dx, dy, math.hypot(dx, dy), minute])
        y.append(int('outcome' not in p))
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50, help='Maximum WC 2018 matches to load, recommended >= 25')
    args = parser.parse_args()
    if args.limit < 8:
        parser.error('Use at least 8 matches for grouped evaluation')
    matches = fetch('matches/43/3.json')
    X, y, groups = [], [], []
    for match in matches[:args.limit]:
        mid = str(match['match_id'])
        try:
            data, labels = parse_match(mid)
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            print(f'WARNING: skipped {mid}: {exc}')
            continue
        X.extend(data)
        y.extend(labels)
        groups.extend([mid] * len(labels))
        print(f'Loaded {mid}: {len(labels)} passes')
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 8 or len(np.unique(y)) != 2:
        raise RuntimeError('Need at least 8 matches and both completed/incomplete passes')
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if len(set(y_train)) != 2 or len(set(y_test)) != 2:
        raise RuntimeError('Train/test sets must contain both outcomes')
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    baseline = DummyClassifier(strategy='prior').fit(X_train, y_train)
    prior_probs = baseline.predict_proba(X_test)[:, 1]
    metrics = {
        'roc_auc': round(float(roc_auc_score(y_test, probabilities)), 4),
        'accuracy': round(float(accuracy_score(y_test, probabilities >= 0.5)), 4),
        'brier': round(float(brier_score_loss(y_test, probabilities)), 4),
        'log_loss': round(float(log_loss(y_test, probabilities)), 4),
        'baseline_accuracy': round(float(accuracy_score(y_test, prior_probs >= 0.5)), 4),
        'baseline_brier': round(float(brier_score_loss(y_test, prior_probs)), 4),
        'baseline_log_loss': round(float(log_loss(y_test, prior_probs)), 4),
        'training_matches': int(len(set(groups[train_idx]))),
        'test_matches': int(len(set(groups[test_idx]))),
        'training_passes': int(len(train_idx)),
        'test_passes': int(len(test_idx)),
    }
    scaler, clf = model.steps[0][1], model.steps[1][1]
    payload = {
        'model_type': 'standardized_logistic_regression',
        'version': 1,
        'trained': True,
        'dataset': 'StatsBomb Open Data, World Cup 2018',
        'task': 'Estimate probability of completed pass given recorded start and endpoint',
        'features': FEATURES,
        'means': scaler.mean_.tolist(),
        'scales': scaler.scale_.tolist(),
        'coefficients': clf.coef_[0].tolist(),
        'intercept': float(clf.intercept_[0]),
        'metrics': metrics,
        'limitations': 'Uses recorded pass endpoint, which may differ from intended target on incomplete passes. Association, not a causal or real-time counterfactual forecast. No defender or pressure features.'
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding='utf8')
    print('\nHeld-out MATCH evaluation:', json.dumps(metrics, indent=2))
    print(f'Exported trained model to {OUT}')
    print('NOTE: Model is not proven useful unless test metrics beat the prior-only baseline.')

if __name__ == '__main__':
    main()
