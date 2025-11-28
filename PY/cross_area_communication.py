"""
Cross-Area Communication Analysis
=================================
Does the communication between brain areas predict behavior?

The Steinmetz dataset is unique - simultaneous recordings from many areas.
Let's see if inter-area correlations/Granger causality predict choice.
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
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'DATA')
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
    """
    Compute cross-correlation between two areas' activity.
    spks1, spks2: (n_neurons, n_trials, n_time)
    Returns: correlation matrix averaged over time window
    """
    t_start, t_end = time_window

    # Average over neurons to get population activity
    pop1 = spks1[:, :, t_start:t_end].mean(axis=0)  # (n_trials, n_time)
    pop2 = spks2[:, :, t_start:t_end].mean(axis=0)

    # Compute correlation per trial
    n_trials = pop1.shape[0]
    corrs = np.zeros(n_trials)

    for trial in range(n_trials):
        if pop1[trial].std() > 0 and pop2[trial].std() > 0:
            corrs[trial] = np.corrcoef(pop1[trial], pop2[trial])[0, 1]
        else:
            corrs[trial] = 0

    return corrs


def compute_granger_proxy(spks1, spks2, time_window=(50, 80)):
    """
    Simple Granger-like causality proxy:
    Does activity in area1 at time t predict activity in area2 at time t+1?

    Returns: correlation of (area1_early, area2_late) per trial
    """
    t_start, t_end = time_window
    t_mid = (t_start + t_end) // 2

    # Early: area1, Late: area2
    pop1_early = spks1[:, :, t_start:t_mid].mean(axis=(0, 2))  # (n_trials,)
    pop2_late = spks2[:, :, t_mid:t_end].mean(axis=(0, 2))

    # Also compute reverse direction
    pop2_early = spks2[:, :, t_start:t_mid].mean(axis=(0, 2))
    pop1_late = spks1[:, :, t_mid:t_end].mean(axis=(0, 2))

    # Return difference: positive = area1 -> area2 dominates
    forward = np.zeros(len(pop1_early))
    backward = np.zeros(len(pop1_early))

    # Simple linear relationship per trial isn't meaningful
    # Instead, compute overall correlation
    r_forward = np.corrcoef(pop1_early, pop2_late)[0, 1] if pop1_early.std() > 0 and pop2_late.std() > 0 else 0
    r_backward = np.corrcoef(pop2_early, pop1_late)[0, 1] if pop2_early.std() > 0 and pop1_late.std() > 0 else 0

    return r_forward, r_backward


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
# Main Analysis: Cross-area correlation predicts choice?
# =============================================================================

print("\n" + "="*70)
print("Analysis 1: Does cross-area correlation predict choice?")
print("="*70)

results_corr = []

for session_idx, session in enumerate(alldat):
    responses = session['response']

    # Only use left/right trials
    valid_trials = responses != 0
    if np.sum(valid_trials) < 50:
        continue

    pairs = get_area_pairs(session, min_neurons=10)
    if len(pairs) < 3:
        continue

    spks = session['spks']
    brain_areas = session['brain_area']

    for area1, area2 in pairs:
        mask1 = brain_areas == area1
        mask2 = brain_areas == area2

        spks1 = spks[mask1]
        spks2 = spks[mask2]

        # Compute cross-correlation
        corrs = compute_cross_correlation(spks1, spks2)

        # Filter valid trials
        X = corrs[valid_trials].reshape(-1, 1)
        y = responses[valid_trials]

        # Convert to binary (left=-1 vs right=1)
        y_binary = (y == 1).astype(int)

        if len(np.unique(y_binary)) < 2:
            continue

        # Logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        try:
            scores = cross_val_score(model, X_scaled, y_binary, cv=5, scoring='accuracy')
            acc = np.mean(scores)

            results_corr.append({
                'session': session_idx,
                'area1': area1,
                'area2': area2,
                'accuracy': acc,
                'n_trials': np.sum(valid_trials)
            })
        except:
            continue

# Summarize
if len(results_corr) > 0:
    import pandas as pd
    df = pd.DataFrame(results_corr)

    print(f"\nTested {len(results_corr)} area pairs across sessions")
    print(f"Mean accuracy: {df['accuracy'].mean():.3f} (chance = 0.5)")
    print(f"Pairs with accuracy > 0.55: {(df['accuracy'] > 0.55).sum()}/{len(df)}")

    # Best pairs
    print("\nTop 10 area pairs:")
    top = df.nlargest(10, 'accuracy')
    print(top[['area1', 'area2', 'accuracy', 'session']].to_string())

    # Statistical test
    t_stat, p_val = stats.ttest_1samp(df['accuracy'].values, 0.5)
    print(f"\nOne-sample t-test (acc > 0.5): t={t_stat:.2f}, p={p_val:.4f}")

    # Group by area pair
    pair_acc = df.groupby(['area1', 'area2'])['accuracy'].agg(['mean', 'std', 'count'])
    pair_acc = pair_acc[pair_acc['count'] >= 3].sort_values('mean', ascending=False)
    print("\nBest area pairs (>= 3 sessions):")
    print(pair_acc.head(10))

else:
    print("No valid results")

# =============================================================================
# Analysis 2: Multi-area features for choice prediction
# =============================================================================

print("\n" + "="*70)
print("Analysis 2: Multi-area activity predicts choice better than single area?")
print("="*70)

results_multi = []

for session_idx, session in enumerate(alldat):
    responses = session['response']
    valid_trials = responses != 0
    if np.sum(valid_trials) < 50:
        continue

    spks = session['spks']
    brain_areas = session['brain_area']

    # Get all areas with enough neurons
    unique_areas = np.unique(brain_areas)
    good_areas = [a for a in unique_areas if np.sum(brain_areas == a) >= 10]

    if len(good_areas) < 2:
        continue

    # Single-area predictions
    single_accs = {}
    for area in good_areas:
        mask = brain_areas == area
        area_spks = spks[mask]

        # Features: mean activity in stimulus period
        X = area_spks[:, :, 50:80].mean(axis=(0, 2))[valid_trials].reshape(-1, 1)
        y = (responses[valid_trials] == 1).astype(int)

        if len(np.unique(y)) < 2:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        try:
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            single_accs[area] = np.mean(scores)
        except:
            continue

    if len(single_accs) < 2:
        continue

    # Multi-area prediction (combine all areas)
    X_multi = []
    for area in good_areas:
        mask = brain_areas == area
        area_spks = spks[mask]
        X_multi.append(area_spks[:, :, 50:80].mean(axis=(0, 2))[valid_trials])

    X_multi = np.column_stack(X_multi)
    y = (responses[valid_trials] == 1).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_multi)

    model = LogisticRegression(max_iter=1000)
    try:
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        multi_acc = np.mean(scores)
    except:
        continue

    # Best single area
    best_single_area = max(single_accs.keys(), key=lambda k: single_accs[k])
    best_single_acc = single_accs[best_single_area]

    results_multi.append({
        'session': session_idx,
        'best_single_acc': best_single_acc,
        'best_single_area': best_single_area,
        'multi_acc': multi_acc,
        'improvement': multi_acc - best_single_acc,
        'n_areas': len(good_areas)
    })

if len(results_multi) > 0:
    df_multi = pd.DataFrame(results_multi)

    print(f"\nTested {len(results_multi)} sessions")
    print(f"Mean best single-area accuracy: {df_multi['best_single_acc'].mean():.3f}")
    print(f"Mean multi-area accuracy: {df_multi['multi_acc'].mean():.3f}")
    print(f"Mean improvement: {df_multi['improvement'].mean():.3f}")

    # Statistical test
    t_stat, p_val = stats.ttest_rel(df_multi['multi_acc'].values, df_multi['best_single_acc'].values)
    print(f"\nPaired t-test (multi > single): t={t_stat:.2f}, p={p_val:.4f}")

    if p_val < 0.05 and df_multi['improvement'].mean() > 0:
        print("\n*** SIGNIFICANT: Multi-area outperforms single area! ***")

# =============================================================================
# Visualization
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Cross-correlation accuracy distribution
if len(results_corr) > 0:
    ax = axes[0]
    ax.hist(df['accuracy'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.axvline(x=df['accuracy'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean={df["accuracy"].mean():.3f}')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Count')
    ax.set_title('Cross-area Correlation\nPredicts Choice?')
    ax.legend()

# Plot 2: Single vs Multi area accuracy
if len(results_multi) > 0:
    ax = axes[1]
    ax.scatter(df_multi['best_single_acc'], df_multi['multi_acc'], alpha=0.6, s=50)
    lims = [0.4, 0.8]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Equal')
    ax.set_xlabel('Best Single-Area Accuracy')
    ax.set_ylabel('Multi-Area Accuracy')
    ax.set_title('Single vs Multi-Area\nChoice Prediction')
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

# Plot 3: Improvement histogram
if len(results_multi) > 0:
    ax = axes[2]
    ax.hist(df_multi['improvement'], bins=15, alpha=0.7, color='forestgreen', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    ax.axvline(x=df_multi['improvement'].mean(), color='blue', linestyle='-', linewidth=2,
               label=f'Mean={df_multi["improvement"].mean():.3f}')
    ax.set_xlabel('Improvement (Multi - Single)')
    ax.set_ylabel('Count')
    ax.set_title('Multi-Area Improvement')
    ax.legend()

plt.tight_layout()
plt.savefig('cross_area_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: cross_area_results.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if len(results_corr) > 0:
    print(f"\n1. Cross-area correlation predicting choice:")
    print(f"   Mean accuracy: {df['accuracy'].mean():.3f} (chance=0.5)")
    if df['accuracy'].mean() > 0.52:
        print("   --> Weak positive effect")

if len(results_multi) > 0:
    print(f"\n2. Multi-area vs single-area prediction:")
    print(f"   Improvement: {df_multi['improvement'].mean():.3f}")
    if df_multi['improvement'].mean() > 0.01:
        print("   --> Multi-area provides additional information!")

print("\nDone!")
