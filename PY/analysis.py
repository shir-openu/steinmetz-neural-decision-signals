import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print('=== Extended Analysis: Multiple Sessions ===')
print()

data_dir = 'DATA'
alldat = []
for i in range(1, 4):
    fname = os.path.join(data_dir, f'steinmetz_part{i}.npy')
    dat = np.load(fname, allow_pickle=True)['dat']
    alldat.extend(dat)

# Find sessions with at least 30 VISp neurons
good_sessions = []
for idx, session in enumerate(alldat):
    visp_count = np.sum(session['brain_area'] == 'VISp')
    n_valid = np.sum(session['response'] != 0)
    if visp_count >= 30 and n_valid >= 100:
        good_sessions.append((idx, visp_count, n_valid))

print(f'Found {len(good_sessions)} sessions with 30+ VISp neurons and 100+ valid trials')
print()

results = []
for session_idx, visp_count, n_valid in good_sessions[:5]:
    session = alldat[session_idx]

    visp_mask = session['brain_area'] == 'VISp'
    spks = session['spks'][visp_mask]
    response = session['response']

    valid_trials = response != 0
    spks_valid = spks[:, valid_trials, :]
    y = (response[valid_trials] == 1).astype(int)

    firing_rates = np.mean(spks_valid[:, :, 5:15], axis=2).T
    n_trials, n_neurons = firing_rates.shape

    max_neurons = min(30, n_neurons)
    firing_rates = firing_rates[:, :max_neurons]
    n_neurons = max_neurons

    firing_rates = (firing_rates - firing_rates.mean(axis=0)) / (firing_rates.std(axis=0) + 1e-6)

    clf = LogisticRegression(max_iter=1000, C=0.1)

    # Baseline
    scores_base = cross_val_score(clf, firing_rates, y, cv=5, scoring='accuracy')

    # Pairwise
    pairwise = []
    for i, j in combinations(range(n_neurons), 2):
        pairwise.append(firing_rates[:, i] * firing_rates[:, j])
    pairwise = np.array(pairwise).T
    X_pair = np.hstack([firing_rates, pairwise])
    scores_pair = cross_val_score(clf, X_pair, y, cv=5, scoring='accuracy')

    # Triplets
    triplet_list = list(combinations(range(n_neurons), 3))
    np.random.seed(42)
    if len(triplet_list) > 500:
        indices = np.random.choice(len(triplet_list), 500, replace=False)
        triplet_list = [triplet_list[i] for i in indices]

    triplets = []
    for i, j, k in triplet_list:
        triplets.append(firing_rates[:, i] * firing_rates[:, j] * firing_rates[:, k])
    triplets = np.array(triplets).T
    X_trip = np.hstack([X_pair, triplets])
    scores_trip = cross_val_score(clf, X_trip, y, cv=5, scoring='accuracy')

    mouse = session['mouse_name']
    print(f'Session {session_idx} ({mouse}): {n_neurons} neurons, {n_trials} trials')
    print(f'  Baseline: {scores_base.mean():.1%}  |  +Pairwise: {scores_pair.mean():.1%}  |  +Triplets: {scores_trip.mean():.1%}')

    results.append({
        'session': session_idx,
        'baseline': scores_base.mean(),
        'pairwise': scores_pair.mean(),
        'triplet': scores_trip.mean()
    })

print()
print('=== Aggregate Results ===')
baseline_avg = np.mean([r['baseline'] for r in results])
pairwise_avg = np.mean([r['pairwise'] for r in results])
triplet_avg = np.mean([r['triplet'] for r in results])

print(f'Mean Baseline:     {baseline_avg:.1%}')
print(f'Mean +Pairwise:    {pairwise_avg:.1%} (delta: {(pairwise_avg-baseline_avg)*100:+.1f}%)')
print(f'Mean +Triplets:    {triplet_avg:.1%} (delta: {(triplet_avg-pairwise_avg)*100:+.1f}%)')
print()

from scipy import stats
pair_vs_base = [r['pairwise'] - r['baseline'] for r in results]
trip_vs_pair = [r['triplet'] - r['pairwise'] for r in results]
_, p_pair = stats.ttest_1samp(pair_vs_base, 0)
_, p_trip = stats.ttest_1samp(trip_vs_pair, 0)
print(f'Pairwise vs Baseline: p = {p_pair:.3f}')
print(f'Triplets vs Pairwise: p = {p_trip:.3f}')
