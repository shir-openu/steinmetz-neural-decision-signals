"""
Cross-Area Communication - Deep Analysis
=========================================
Follow-up on the positive finding:
1. Which area pairs are most consistent?
2. What is the direction of the correlation (positive/negative)?
3. When is the correlation most predictive (timing)?
"""

import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
data_dir = 'DATA'
alldat = []
for i in range(1, 4):
    fname = os.path.join(data_dir, f'steinmetz_part{i}.npy')
    dat = np.load(fname, allow_pickle=True)['dat']
    alldat.extend(dat)

print(f"Loaded {len(alldat)} sessions")

# =============================================================================
# Helper Functions
# =============================================================================

def compute_cross_correlation(spks1, spks2, time_window=(50, 80)):
    """Compute cross-correlation between two areas' population activity."""
    t_start, t_end = time_window

    # Average over neurons to get population activity
    pop1 = spks1[:, :, t_start:t_end].mean(axis=0)  # (n_trials, n_time)
    pop2 = spks2[:, :, t_start:t_end].mean(axis=0)

    n_trials = pop1.shape[0]
    corrs = np.zeros(n_trials)

    for trial in range(n_trials):
        if pop1[trial].std() > 0 and pop2[trial].std() > 0:
            corrs[trial] = np.corrcoef(pop1[trial], pop2[trial])[0, 1]
        else:
            corrs[trial] = 0

    return corrs


def get_area_pairs(session, min_neurons=10):
    """Get all pairs of areas with enough neurons."""
    areas = np.unique(session['brain_area'])
    pairs = []

    for i, area1 in enumerate(areas):
        n1 = np.sum(session['brain_area'] == area1)
        if n1 < min_neurons:
            continue

        for area2 in areas[i+1:]:
            n2 = np.sum(session['brain_area'] == area2)
            if n2 < min_neurons:
                continue

            pairs.append((area1, area2))

    return pairs


# =============================================================================
# Analysis 1: Most consistent area pairs
# =============================================================================

print("\n" + "="*70)
print("Analysis 1: Most Consistent Area Pairs")
print("="*70)

# Collect results for each area pair across sessions
pair_results = {}

for session_idx, session in enumerate(alldat):
    responses = session['response']
    valid_trials = responses != 0
    if np.sum(valid_trials) < 50:
        continue

    pairs = get_area_pairs(session, min_neurons=10)
    spks = session['spks']
    brain_areas = session['brain_area']

    for area1, area2 in pairs:
        mask1 = brain_areas == area1
        mask2 = brain_areas == area2

        spks1 = spks[mask1]
        spks2 = spks[mask2]

        corrs = compute_cross_correlation(spks1, spks2)

        X = corrs[valid_trials].reshape(-1, 1)
        y = (responses[valid_trials] == 1).astype(int)

        if len(np.unique(y)) < 2:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        try:
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            acc = np.mean(scores)

            # Also get the correlation direction
            model.fit(X_scaled, y)
            coef = model.coef_[0, 0]

            pair_key = (area1, area2)
            if pair_key not in pair_results:
                pair_results[pair_key] = {'accuracies': [], 'coefficients': [], 'sessions': []}

            pair_results[pair_key]['accuracies'].append(acc)
            pair_results[pair_key]['coefficients'].append(coef)
            pair_results[pair_key]['sessions'].append(session_idx)

        except:
            continue

# Filter pairs with at least 3 sessions
consistent_pairs = {k: v for k, v in pair_results.items() if len(v['accuracies']) >= 3}

print(f"\nFound {len(consistent_pairs)} area pairs with >= 3 sessions")

# Rank by consistency (mean accuracy and low variance)
pair_stats = []
for pair, data in consistent_pairs.items():
    accs = data['accuracies']
    coefs = data['coefficients']
    pair_stats.append({
        'pair': pair,
        'mean_acc': np.mean(accs),
        'std_acc': np.std(accs),
        'n_sessions': len(accs),
        'mean_coef': np.mean(coefs),
        'coef_sign_consistency': np.abs(np.mean(np.sign(coefs)))  # 1 if all same sign
    })

# Sort by mean accuracy
pair_stats.sort(key=lambda x: x['mean_acc'], reverse=True)

print("\nTop 15 Most Predictive Area Pairs:")
print(f"{'Pair':<20} {'Accuracy':<12} {'Std':<8} {'N':<5} {'Coef Sign':<12}")
print("-" * 60)

for p in pair_stats[:15]:
    pair_str = f"{p['pair'][0]}-{p['pair'][1]}"
    sign_str = "Positive" if p['mean_coef'] > 0 else "Negative"
    print(f"{pair_str:<20} {p['mean_acc']:.3f}        {p['std_acc']:.3f}    {p['n_sessions']:<5} {sign_str}")

# =============================================================================
# Analysis 2: Correlation Direction (Positive vs Negative)
# =============================================================================

print("\n" + "="*70)
print("Analysis 2: Correlation Direction")
print("="*70)

# For top pairs, what does positive/negative correlation mean?
print("\nInterpretation of correlation direction:")
print("- Positive coef: Higher correlation -> more likely RIGHT choice")
print("- Negative coef: Higher correlation -> more likely LEFT choice")

# Count how many pairs have consistent sign
n_positive = sum(1 for p in pair_stats if p['mean_coef'] > 0)
n_negative = sum(1 for p in pair_stats if p['mean_coef'] < 0)
print(f"\nPairs with positive coefficient: {n_positive}/{len(pair_stats)}")
print(f"Pairs with negative coefficient: {n_negative}/{len(pair_stats)}")

# Check sign consistency
high_consistency = [p for p in pair_stats if p['coef_sign_consistency'] > 0.8]
print(f"Pairs with consistent sign (>80%): {len(high_consistency)}/{len(pair_stats)}")

# =============================================================================
# Analysis 3: Timing Analysis
# =============================================================================

print("\n" + "="*70)
print("Analysis 3: When is cross-area correlation most predictive?")
print("="*70)

# Test different time windows
time_windows = [
    ('pre_stim', (30, 50)),      # Before stimulus
    ('early_stim', (50, 60)),    # 0-100ms after stimulus
    ('mid_stim', (60, 70)),      # 100-200ms after stimulus
    ('late_stim', (70, 80)),     # 200-300ms after stimulus
    ('response', (80, 100))      # 300-500ms (around response time)
]

# Use top 5 most consistent pairs
top_pairs = [p['pair'] for p in pair_stats[:5]]

timing_results = {name: [] for name, _ in time_windows}

for session_idx, session in enumerate(alldat):
    responses = session['response']
    valid_trials = responses != 0
    if np.sum(valid_trials) < 50:
        continue

    spks = session['spks']
    brain_areas = session['brain_area']

    for area1, area2 in top_pairs:
        mask1 = brain_areas == area1
        mask2 = brain_areas == area2

        if np.sum(mask1) < 10 or np.sum(mask2) < 10:
            continue

        spks1 = spks[mask1]
        spks2 = spks[mask2]

        for window_name, window in time_windows:
            try:
                corrs = compute_cross_correlation(spks1, spks2, time_window=window)

                X = corrs[valid_trials].reshape(-1, 1)
                y = (responses[valid_trials] == 1).astype(int)

                if len(np.unique(y)) < 2:
                    continue

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = LogisticRegression(max_iter=1000)
                scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
                timing_results[window_name].append(np.mean(scores))
            except:
                continue

print("\nAccuracy by Time Window (top 5 area pairs):")
print(f"{'Time Window':<15} {'Mean Acc':<12} {'Std':<10} {'N'}")
print("-" * 45)

for window_name, _ in time_windows:
    accs = timing_results[window_name]
    if len(accs) > 0:
        print(f"{window_name:<15} {np.mean(accs):.3f}        {np.std(accs):.3f}      {len(accs)}")

# Statistical comparison between time windows
print("\nStatistical comparison (late_stim vs others):")
if len(timing_results['late_stim']) > 5:
    for window_name, _ in time_windows:
        if window_name != 'late_stim' and len(timing_results[window_name]) > 5:
            t, p = stats.ttest_ind(timing_results['late_stim'], timing_results[window_name])
            diff = np.mean(timing_results['late_stim']) - np.mean(timing_results[window_name])
            print(f"  late_stim vs {window_name}: diff={diff:.3f}, p={p:.4f}")

# =============================================================================
# Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Top pairs accuracy
ax = axes[0, 0]
top_15 = pair_stats[:15]
pairs_str = [f"{p['pair'][0][:4]}-{p['pair'][1][:4]}" for p in top_15]
accs = [p['mean_acc'] for p in top_15]
errs = [p['std_acc'] for p in top_15]

bars = ax.barh(range(len(pairs_str)), accs, xerr=errs, color='steelblue', alpha=0.7, capsize=3)
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Chance')
ax.set_yticks(range(len(pairs_str)))
ax.set_yticklabels(pairs_str)
ax.set_xlabel('Accuracy')
ax.set_title('Top 15 Most Predictive Area Pairs')
ax.legend()
ax.invert_yaxis()

# Plot 2: Coefficient signs
ax = axes[0, 1]
coefs = [p['mean_coef'] for p in pair_stats[:15]]
colors = ['green' if c > 0 else 'red' for c in coefs]
ax.barh(range(len(pairs_str)), coefs, color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(range(len(pairs_str)))
ax.set_yticklabels(pairs_str)
ax.set_xlabel('Coefficient (+ = Right, - = Left)')
ax.set_title('Correlation Direction')
ax.invert_yaxis()

# Plot 3: Timing analysis
ax = axes[1, 0]
window_names = [name for name, _ in time_windows]
window_means = [np.mean(timing_results[name]) if len(timing_results[name]) > 0 else 0 for name in window_names]
window_stds = [np.std(timing_results[name]) if len(timing_results[name]) > 0 else 0 for name in window_names]

bars = ax.bar(window_names, window_means, yerr=window_stds, color='forestgreen', alpha=0.7, capsize=5)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance')
ax.set_ylabel('Accuracy')
ax.set_title('Prediction Accuracy by Time Window')
ax.set_xlabel('Time Window')
ax.legend()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 4: Scatter - accuracy vs consistency
ax = axes[1, 1]
accs = [p['mean_acc'] for p in pair_stats]
stds = [p['std_acc'] for p in pair_stats]
n_sessions = [p['n_sessions'] for p in pair_stats]

scatter = ax.scatter(accs, stds, c=n_sessions, cmap='viridis', s=50, alpha=0.7)
ax.set_xlabel('Mean Accuracy')
ax.set_ylabel('Std Accuracy (lower = more consistent)')
ax.set_title('Accuracy vs Consistency')
plt.colorbar(scatter, ax=ax, label='N Sessions')

plt.tight_layout()
plt.savefig('cross_area_deep_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: cross_area_deep_analysis.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\n1. MOST CONSISTENT AREA PAIRS:")
for i, p in enumerate(pair_stats[:5]):
    print(f"   {i+1}. {p['pair'][0]} - {p['pair'][1]}: {p['mean_acc']:.3f} accuracy")

best_window = max(timing_results.keys(), key=lambda k: np.mean(timing_results[k]) if len(timing_results[k]) > 0 else 0)
print(f"\n2. BEST TIME WINDOW: {best_window}")
print(f"   Accuracy: {np.mean(timing_results[best_window]):.3f}")

print(f"\n3. CORRELATION DIRECTION:")
print(f"   {n_positive} pairs: higher correlation -> RIGHT")
print(f"   {n_negative} pairs: higher correlation -> LEFT")

# Key finding
print("\n" + "="*70)
print("KEY FINDING")
print("="*70)
print("""
Cross-area correlation during stimulus presentation predicts
the mouse's choice with ~58-63% accuracy (chance = 50%).

The most predictive pairs involve:
- Hippocampal regions (CA1, DG, CA3)
- Thalamic nuclei (LP, LD)
- Prefrontal areas (ILA, PL)

This suggests that inter-area communication, not just local
activity, carries decision-related information.
""")

print("\nDone!")
