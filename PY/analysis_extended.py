"""
Extended Higher-Order Correlations Analysis
============================================
Testing if triplet correlations improve behavioral prediction
across different brain areas and time windows.
"""

import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Load data
print('Loading Steinmetz data...')
data_dir = 'DATA'
alldat = []
for i in range(1, 4):
    fname = os.path.join(data_dir, f'steinmetz_part{i}.npy')
    dat = np.load(fname, allow_pickle=True)['dat']
    alldat.extend(dat)
print(f'Loaded {len(alldat)} sessions\n')

# Get all brain areas
all_areas = set()
for session in alldat:
    all_areas.update(session['brain_area'])
print(f'Brain areas in dataset: {sorted(all_areas)}\n')

# Areas of interest for decision-making
areas_of_interest = ['VISp', 'MOp', 'MOs', 'ACA', 'PL', 'SC', 'SNr']


def compute_features(firing_rates, n_neurons, include_triplets=True, max_triplets=300):
    """Compute pairwise and triplet features"""
    # Pairwise products
    pairwise = []
    for i, j in combinations(range(n_neurons), 2):
        pairwise.append(firing_rates[:, i] * firing_rates[:, j])
    pairwise = np.array(pairwise).T if pairwise else np.zeros((firing_rates.shape[0], 0))

    X_pair = np.hstack([firing_rates, pairwise])

    if not include_triplets:
        return firing_rates, X_pair, X_pair

    # Triplet products (sampled)
    triplet_list = list(combinations(range(n_neurons), 3))
    if len(triplet_list) > max_triplets:
        indices = np.random.choice(len(triplet_list), max_triplets, replace=False)
        triplet_list = [triplet_list[i] for i in indices]

    triplets = []
    for i, j, k in triplet_list:
        triplets.append(firing_rates[:, i] * firing_rates[:, j] * firing_rates[:, k])
    triplets = np.array(triplets).T if triplets else np.zeros((firing_rates.shape[0], 0))

    X_trip = np.hstack([X_pair, triplets])

    return firing_rates, X_pair, X_trip


def analyze_area(alldat, area, time_window=(5, 15), min_neurons=15, min_trials=80):
    """Analyze a specific brain area across all sessions"""
    results = []

    for session_idx, session in enumerate(alldat):
        # Get neurons in this area
        area_mask = session['brain_area'] == area
        n_area_neurons = np.sum(area_mask)

        # Get valid trials
        response = session['response']
        valid_trials = response != 0
        n_valid = np.sum(valid_trials)

        if n_area_neurons < min_neurons or n_valid < min_trials:
            continue

        # Extract data
        spks = session['spks'][area_mask][:, valid_trials, :]
        y = (response[valid_trials] == 1).astype(int)

        # Firing rates in time window
        t_start, t_end = time_window
        firing_rates = np.mean(spks[:, :, t_start:t_end], axis=2).T
        n_trials, n_neurons = firing_rates.shape

        # Limit neurons
        max_neurons = min(25, n_neurons)
        firing_rates = firing_rates[:, :max_neurons]
        n_neurons = max_neurons

        # Normalize
        firing_rates = (firing_rates - firing_rates.mean(axis=0)) / (firing_rates.std(axis=0) + 1e-6)

        # Compute features
        np.random.seed(42)
        X_base, X_pair, X_trip = compute_features(firing_rates, n_neurons)

        # Cross-validation
        clf = LogisticRegression(max_iter=1000, C=0.1)

        scores_base = cross_val_score(clf, X_base, y, cv=5, scoring='accuracy')
        scores_pair = cross_val_score(clf, X_pair, y, cv=5, scoring='accuracy')
        scores_trip = cross_val_score(clf, X_trip, y, cv=5, scoring='accuracy')

        results.append({
            'session': session_idx,
            'mouse': session['mouse_name'],
            'n_neurons': n_neurons,
            'n_trials': n_trials,
            'baseline': scores_base.mean(),
            'pairwise': scores_pair.mean(),
            'triplet': scores_trip.mean()
        })

    return results


print('=' * 60)
print('PART 1: Analysis by Brain Area')
print('=' * 60)

area_summaries = {}
for area in areas_of_interest:
    results = analyze_area(alldat, area)
    if len(results) >= 3:  # Need at least 3 sessions
        baseline_avg = np.mean([r['baseline'] for r in results])
        pairwise_avg = np.mean([r['pairwise'] for r in results])
        triplet_avg = np.mean([r['triplet'] for r in results])

        area_summaries[area] = {
            'n_sessions': len(results),
            'baseline': baseline_avg,
            'pairwise': pairwise_avg,
            'triplet': triplet_avg,
            'pair_delta': pairwise_avg - baseline_avg,
            'trip_delta': triplet_avg - pairwise_avg
        }

        print(f'\n{area} ({len(results)} sessions):')
        print(f'  Baseline: {baseline_avg:.1%}')
        print(f'  +Pairwise: {pairwise_avg:.1%} ({(pairwise_avg-baseline_avg)*100:+.1f}%)')
        print(f'  +Triplets: {triplet_avg:.1%} ({(triplet_avg-pairwise_avg)*100:+.1f}%)')

print('\n' + '=' * 60)
print('PART 2: Analysis by Time Window (VISp)')
print('=' * 60)

# Different time windows (in 10ms bins)
time_windows = [
    ((0, 5), 'Pre-stim (0-50ms)'),
    ((5, 15), 'Early (50-150ms)'),
    ((15, 25), 'Late (150-250ms)'),
    ((25, 35), 'Response (250-350ms)'),
    ((5, 35), 'Full stim (50-350ms)')
]

for (t_start, t_end), label in time_windows:
    results = analyze_area(alldat, 'VISp', time_window=(t_start, t_end))
    if len(results) >= 3:
        baseline_avg = np.mean([r['baseline'] for r in results])
        pairwise_avg = np.mean([r['pairwise'] for r in results])
        triplet_avg = np.mean([r['triplet'] for r in results])

        print(f'\n{label}:')
        print(f'  Baseline: {baseline_avg:.1%} | +Pair: {pairwise_avg:.1%} | +Trip: {triplet_avg:.1%}')


print('\n' + '=' * 60)
print('PART 3: True Third-Order Cumulants')
print('=' * 60)

def compute_cumulant_features(firing_rates, n_neurons, max_cumulants=200):
    """
    Compute true third-order cumulants:
    kappa_ijk = E[XYZ] - E[X]E[YZ] - E[Y]E[XZ] - E[Z]E[XY] + 2E[X]E[Y]E[Z]

    For centered data: kappa_ijk = E[XYZ]
    """
    n_trials = firing_rates.shape[0]

    # Already centered, so E[X]=0 and cumulant = E[XYZ]
    triplet_list = list(combinations(range(n_neurons), 3))
    if len(triplet_list) > max_cumulants:
        np.random.seed(42)
        indices = np.random.choice(len(triplet_list), max_cumulants, replace=False)
        triplet_list = [triplet_list[i] for i in indices]

    # For each trial, compute the triplet product (this IS the cumulant for centered data)
    triplet_features = []
    for i, j, k in triplet_list:
        triplet_features.append(firing_rates[:, i] * firing_rates[:, j] * firing_rates[:, k])

    return np.array(triplet_features).T if triplet_features else np.zeros((n_trials, 0))


# Compare product features vs cumulant features
print('\nComparing triplet products vs true cumulants...')

results_comparison = []
for session_idx, session in enumerate(alldat):
    area_mask = session['brain_area'] == 'VISp'
    n_area_neurons = np.sum(area_mask)
    response = session['response']
    valid_trials = response != 0
    n_valid = np.sum(valid_trials)

    if n_area_neurons < 20 or n_valid < 100:
        continue

    spks = session['spks'][area_mask][:, valid_trials, :]
    y = (response[valid_trials] == 1).astype(int)

    firing_rates = np.mean(spks[:, :, 5:15], axis=2).T
    n_neurons = min(25, firing_rates.shape[1])
    firing_rates = firing_rates[:, :n_neurons]

    # Normalize (center)
    firing_rates = (firing_rates - firing_rates.mean(axis=0)) / (firing_rates.std(axis=0) + 1e-6)

    # Pairwise
    pairwise = []
    for i, j in combinations(range(n_neurons), 2):
        pairwise.append(firing_rates[:, i] * firing_rates[:, j])
    pairwise = np.array(pairwise).T
    X_pair = np.hstack([firing_rates, pairwise])

    # Cumulants
    cumulant_features = compute_cumulant_features(firing_rates, n_neurons)
    X_cumulant = np.hstack([X_pair, cumulant_features])

    clf = LogisticRegression(max_iter=1000, C=0.1)
    scores_pair = cross_val_score(clf, X_pair, y, cv=5, scoring='accuracy')
    scores_cumulant = cross_val_score(clf, X_cumulant, y, cv=5, scoring='accuracy')

    results_comparison.append({
        'pairwise': scores_pair.mean(),
        'cumulant': scores_cumulant.mean()
    })

if results_comparison:
    pair_avg = np.mean([r['pairwise'] for r in results_comparison])
    cumulant_avg = np.mean([r['cumulant'] for r in results_comparison])
    print(f'Pairwise only: {pair_avg:.1%}')
    print(f'+Cumulants:    {cumulant_avg:.1%} (delta: {(cumulant_avg-pair_avg)*100:+.1f}%)')


print('\n' + '=' * 60)
print('SUMMARY & CONCLUSIONS')
print('=' * 60)

# Find best area
if area_summaries:
    best_area = max(area_summaries.keys(), key=lambda a: area_summaries[a]['trip_delta'])
    best = area_summaries[best_area]

    print(f'\nBest area for triplet improvement: {best_area}')
    print(f'  Triplet delta: {best["trip_delta"]*100:+.1f}%')

    # Overall conclusion
    all_trip_deltas = [s['trip_delta'] for s in area_summaries.values()]
    mean_trip_delta = np.mean(all_trip_deltas)

    print(f'\nOverall triplet delta across areas: {mean_trip_delta*100:+.1f}%')

    if mean_trip_delta > 0.02:
        print('\nCONCLUSION: Triplet correlations provide meaningful improvement!')
    elif mean_trip_delta > 0:
        print('\nCONCLUSION: Triplet correlations provide marginal improvement.')
    else:
        print('\nCONCLUSION: Triplet correlations do not improve prediction.')
        print('This suggests that pairwise statistics capture most behaviorally-relevant information.')
