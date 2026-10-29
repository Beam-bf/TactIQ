"""Train and evaluate TactiQ AI pass-completion classifier.

From project root:
  python -m pip install numpy scikit-learn
  python analytics/train_pass_model.py --limit 50

Exports frontend/public/pass-model.json and frontend/public/model-evaluation.json.
Uses StatsBomb 2018 World Cup open events.
"""
import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = 'https://raw.githubusercontent.com/hudl/open-data/master/data'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'frontend' / 'public' / 'pass-model.json'
EVAL_OUT = ROOT / 'frontend' / 'public' / 'model-evaluation.json'
FEATURES = ['x', 'y', 'dx', 'dy', 'distance', 'minute']
def evaluate_thresholds(y_true, probabilities):
    thresholds = [0.5, 0.6, 0.7, 0.8]
    results = []

    for threshold in thresholds:
        # Predict incomplete when completion probability
        # is below the threshold.
        predicted_incomplete = probabilities < threshold
        actual_incomplete = y_true == 0

        results.append({
            "threshold": threshold,
            "incomplete_precision": round(
                precision_score(
                    actual_incomplete,
                    predicted_incomplete,
                    zero_division=0
                ), 4
            ),
            "incomplete_recall": round(
                recall_score(
                    actual_incomplete,
                    predicted_incomplete,
                    zero_division=0
                ), 4
            ),
            "incomplete_f1": round(
                f1_score(
                    actual_incomplete,
                    predicted_incomplete,
                    zero_division=0
                ), 4
            )
        })

    return results

def fetch(path):
    req = Request(f'{BASE}/{path}', headers={'User-Agent': 'TactiQ-AI-academic-analytics/1.0'})
    with urlopen(req, timeout=45) as response:
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
        minute = float(event.get('minute', 0))
        if not math.isfinite(minute):
            continue
        dx, dy = ex - x, ey - side
        X.append([x, side, dx, dy, math.hypot(dx, dy), minute])
        # StatsBomb's null outcome on passes normally represents completion.
        y.append(int('outcome' not in p))
    return X, y


def safe_auc(y, probs):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, probs))


def metrics_for(y, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, preds, labels=[0, 1])

    return {
        'accuracy': float(accuracy_score(y, preds)),
        'brier': float(brier_score_loss(y, probs)),
        'log_loss': float(log_loss(y, probs, labels=[0, 1])),
        'roc_auc': safe_auc(y, probs),
        'incomplete_precision': float(precision_score(y, preds, pos_label=0, zero_division=0)),
        'incomplete_recall': float(recall_score(y, preds, pos_label=0, zero_division=0)),
        'incomplete_f1': float(f1_score(y, preds, pos_label=0, zero_division=0)),
        'completed_f1': float(f1_score(y, preds, pos_label=1, zero_division=0)),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': ['incomplete', 'completed'],
    }


def reliability_bins(y, probs, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    output = []
    for k in range(bins):
        mask = (probs >= edges[k]) & (probs < edges[k+1] if k < bins-1 else probs <= edges[k+1])
        if not np.any(mask):
            continue
        output.append({
            'lower': float(edges[k]),
            'upper': float(edges[k+1]),
            'count': int(np.sum(mask)),
            'mean_predicted': float(np.mean(probs[mask])),
            'observed_completion': float(np.mean(y[mask])),
        })
    return output


def match_bootstrap(y, probs, base_probs, groups, reps=1000, seed=42):
    """Resample held-out matches, not individual pass events."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == g) for g in unique]
    changes = []
    for _ in range(reps):
        selected = rng.integers(0, len(indices), size=len(indices))
        idx = np.concatenate([indices[i] for i in selected])
        # Positive means our model improves Brier score vs constant prior.
        changes.append(float(brier_score_loss(y[idx], base_probs[idx]) - brier_score_loss(y[idx], probs[idx])))
    return {
        'unit': 'held-out match',
        'resamples': reps,
        'brier_improvement_model_vs_prior': {
            'lower_95': float(np.percentile(changes, 2.5)),
            'upper_95': float(np.percentile(changes, 97.5)),
        },
    }


def rounded(obj):
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 5)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--bootstrap', type=int, default=1000, help='Match bootstrap resamples')
    args = parser.parse_args()
    if args.limit < 8:
        parser.error('Use at least 8 matches for grouped evaluation')
    if args.bootstrap < 100:
        parser.error('Use at least 100 bootstrap resamples')

    matches = fetch('matches/43/3.json')
    X, y, groups = [], [], []
    for match in matches[:args.limit]:
        mid = str(match['match_id'])
        try:
            data, labels = parse_match(mid)
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            print(f'WARNING: skipped {mid}: {exc}')
            continue
        X.extend(data)
        y.extend(labels)
        groups.extend([mid] * len(labels))
        print(f'Loaded {mid}: {len(labels)} passes')

    X, y, groups = np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(groups)
    if len(np.unique(groups)) < 8 or len(np.unique(y)) != 2:
        raise RuntimeError('Need at least 8 matches and both outcomes')
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise RuntimeError('Train and test need both outcomes')
                    # Split training matches into training and validation matches
    validation_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=123
    )

    subtrain_idx, val_idx = next(
        validation_split.split(
            X_train,
            y_train,
            groups[train_idx]
        )
    )

    X_subtrain = X_train[subtrain_idx]
    y_subtrain = y_train[subtrain_idx]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    print("Training passes:", len(X_subtrain))
    print("Validation passes:", len(X_val))
    print("Test passes:", len(X_test))
    
    # Fit a temporary model using ONLY subtraining matches.
    # Validation matches select the threshold; held-out test matches remain unseen.
    validation_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
    )
    validation_model.fit(X_subtrain, y_subtrain)
    validation_probs = validation_model.predict_proba(X_val)[:, 1]
    threshold_results = evaluate_thresholds(y_val, validation_probs)
    best_threshold = max(
        threshold_results,
        key=lambda result: (result['incomplete_f1'], -result['threshold'])
    )['threshold']
    print('\nVALIDATION THRESHOLD COMPARISON')
    for row in threshold_results:
        print(row)
    print('Selected threshold:', best_threshold)

    # Refit on all non-test matches after selecting the threshold.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    prior_model = DummyClassifier(strategy='prior').fit(X_train, y_train)
    prior_probs = prior_model.predict_proba(X_test)[:, 1]
    always_complete_probs = np.ones_like(probs)
    results = metrics_for(y_test, probs, threshold=best_threshold)

    baseline = metrics_for(y_test, prior_probs, threshold=0.5)
    always_complete = {
        'accuracy': float(accuracy_score(y_test, np.ones_like(y_test))),
        'incomplete_recall': 0.0,
        'note': 'Always predicts completed; Brier/log-loss for hard 100% probabilities are not used as calibration benchmarks.'
    }
    calibration = reliability_bins(y_test, probs)
    bootstrap = match_bootstrap(y_test, probs, prior_probs, groups[test_idx], reps=args.bootstrap)

    # Preserve original dashboard-compatible metric keys.
    metrics = {
        'roc_auc': results['roc_auc'],
        'accuracy': results['accuracy'],
        'brier': results['brier'],
        'log_loss': results['log_loss'],
        'baseline_accuracy': baseline['accuracy'],
        'baseline_brier': baseline['brier'],
        'baseline_log_loss': baseline['log_loss'],
        'training_matches': int(len(np.unique(groups[train_idx]))),
        'test_matches': int(len(np.unique(groups[test_idx]))),
        'training_passes': int(len(train_idx)),
        'test_passes': int(len(test_idx)),
        'incomplete_precision': results['incomplete_precision'],
        'incomplete_recall': results['incomplete_recall'],
        'incomplete_f1': results['incomplete_f1'],
        'always_complete_accuracy': always_complete['accuracy'],
        'selected_threshold': best_threshold,
        'validation_matches': int(len(np.unique(groups[train_idx][val_idx]))),
    }
    scaler, clf = model.steps[0][1], model.steps[1][1]
    payload = {
        'model_type': 'standardized_logistic_regression',
        'version': 3,
        'trained': True,
        'dataset': 'StatsBomb Open Data, World Cup 2018',
        'task': 'Estimate completion given recorded start and endpoint of pass',
        'features': FEATURES,
        'means': scaler.mean_.tolist(),
        'scales': scaler.scale_.tolist(),
        'coefficients': clf.coef_[0].tolist(),
        'intercept': float(clf.intercept_[0]),
        'metrics': rounded(metrics),
        'limitations': ('Observed endpoint on incomplete passes may not equal intended target. '
                        'This is retrospective association, not a real-time recommendation or causal simulation. '
                        'No defender pressure, time-of-pass game state, or tracking features.'),
    }
    evaluation = {
        'dataset': payload['dataset'],
        'test_match_count': metrics['test_matches'],
        'test_pass_count': metrics['test_passes'],
        'positive_label': 'completed',
        'selected_threshold': best_threshold,
        'validation_threshold_results': threshold_results,
        'results': rounded(results),
        'constant_prior_baseline': rounded(baseline),
        'always_complete_baseline': rounded(always_complete),
        'calibration_bins': rounded(calibration),
        'match_bootstrap': rounded(bootstrap),
        'limitations': payload['limitations'],
        'note': ('Confusion matrix rows=actual, columns=predicted, in label order '
                 '[incomplete, completed]. Single grouped holdout split; not proof of transfer to other competitions.'),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    EVAL_OUT.write_text(json.dumps(evaluation, indent=2), encoding='utf-8')
    print('\nHELD-OUT MODEL EVALUATION')
    print(json.dumps(evaluation, indent=2))
    print(f'\nExported {OUT}')
    print(f'Exported {EVAL_OUT}')
    print('Check incomplete recall, calibration, and match-bootstrap interval before pitching.')


if __name__ == '__main__':
    main()
