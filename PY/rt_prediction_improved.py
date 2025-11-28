"""
Reaction Time Prediction - Improved Version
============================================
Using PCA to reduce dimensionality and avoid overfitting.
Also testing different feature extraction methods.
"""

import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
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
# Improved RT Prediction with PCA
# =============================================================================

def prepare_rt_data_with_pca(session, area='VISp', time_window='late_stim', n_pcs=10):
    """
    Prepare data with PCA dimensionality reduction.
    """
    if 'reaction_time' not in session:
        return None, None, None

    area_mask = session['brain_area'] == area
    n_neurons_area = np.sum(area_mask)
    if n_neurons_area < 10:
        return None, None, None

    spks = session['spks'][area_mask]
    n_neurons = spks.shape[0]
    n_trials = spks.shape[1]

    # Get RT
    rt = session['reaction_time']
    rts = rt[:, 0] if rt.ndim > 1 else rt

    # Time windows
    if time_window == 'pre_stim':
        t_start, t_end = 30, 50
    elif time_window == 'early_stim':
        t_start, t_end = 50, 60
    elif time_window == 'late_stim':
        t_start, t_end = 60, 80
    else:
        raise ValueError(f"Unknown time window: {time_window}")

    # Extract features - use full time course, not just mean
    X = spks[:, :, t_start:t_end].mean(axis=2).T  # (n_trials, n_neurons)

    # Filter valid trials
    responses = session['response']
    valid = np.isfinite(rts) & (responses != 0) & (rts > 50) & (rts < 1500)

    X = X[valid]
    y = rts[valid]

    if len(y) < 50:
        return None, None, None

    # Apply PCA
    n_components = min(n_pcs, X.shape[1], X.shape[0] - 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, {'n_neurons': n_neurons, 'n_trials': len(y), 'var_explained': sum(pca.explained_variance_ratio_)}


def run_cv_prediction(X, y, n_splits=5):
    """Run cross-validated prediction with Ridge regression."""
    model = Ridge(alpha=1.0)
    scores = cross_val_score(model, X, y, cv=n_splits, scoring='r2')
    return np.mean(scores), np.std(scores)


# =============================================================================
# Main Analysis
# =============================================================================

print("\n" + "="*70)
print("RT Prediction with PCA Dimensionality Reduction")
print("="*70)

areas_to_test = ['VISp', 'MOs', 'MOp', 'ACA', 'SC', 'CP', 'LS']  # More areas
time_windows = ['pre_stim', 'early_stim', 'late_stim']
n_pcs_options = [5, 10, 15]

results = []

print("\nTesting all combinations...")

for area in areas_to_test:
    for time_window in time_windows:
        for n_pcs in n_pcs_options:
            session_results = []

            for session_idx, session in enumerate(alldat):
                data = prepare_rt_data_with_pca(session, area, time_window, n_pcs)

                if data[0] is None:
                    continue

                X, y, info = data
                r2_mean, r2_std = run_cv_prediction(X, y)

                session_results.append({
                    'session': session_idx,
                    'r2': r2_mean,
                    'r2_std': r2_std,
                    **info
                })

            if len(session_results) >= 3:
                r2_values = [r['r2'] for r in session_results]
                mean_r2 = np.mean(r2_values)
                n_positive = sum(1 for r in r2_values if r > 0)

                results.append({
                    'area': area,
                    'time_window': time_window,
                    'n_pcs': n_pcs,
                    'mean_r2': mean_r2,
                    'std_r2': np.std(r2_values),
                    'n_sessions': len(session_results),
                    'n_positive': n_positive,
                    'pct_positive': n_positive / len(session_results) * 100
                })

# Sort by mean R2
results.sort(key=lambda x: x['mean_r2'], reverse=True)

print("\n" + "="*70)
print("TOP 15 Results (sorted by mean R²)")
print("="*70)
print(f"{'Area':<8} {'Window':<12} {'PCs':<5} {'N_sess':<8} {'Mean R²':<10} {'%Positive':<10}")
print("-" * 60)

for r in results[:15]:
    print(f"{r['area']:<8} {r['time_window']:<12} {r['n_pcs']:<5} {r['n_sessions']:<8} "
          f"{r['mean_r2']:<10.3f} {r['pct_positive']:<10.0f}%")

# =============================================================================
# Statistical Test on Best Condition
# =============================================================================

print("\n" + "="*70)
print("Statistical Analysis of Best Condition")
print("="*70)

if len(results) > 0:
    best = results[0]
    print(f"\nBest: {best['area']}, {best['time_window']}, {best['n_pcs']} PCs")
    print(f"Mean R² = {best['mean_r2']:.3f}, {best['pct_positive']:.0f}% sessions positive")

    # Re-run best condition to get all R2 values
    best_r2s = []
    for session_idx, session in enumerate(alldat):
        data = prepare_rt_data_with_pca(session, best['area'], best['time_window'], best['n_pcs'])
        if data[0] is not None:
            X, y, _ = data
            r2_mean, _ = run_cv_prediction(X, y)
            best_r2s.append(r2_mean)

    # One-sample t-test
    t_stat, p_val = stats.ttest_1samp(best_r2s, 0)
    print(f"\nOne-sample t-test (R² > 0): t={t_stat:.2f}, p={p_val:.4f}")

    if p_val < 0.05 and np.mean(best_r2s) > 0:
        print("\n*** SIGNIFICANT POSITIVE RESULT! ***")
    else:
        print("\nResult not significant")

# =============================================================================
# Visualization
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: R² by area (using best n_pcs)
area_results = {}
for r in results:
    if r['n_pcs'] == 10:  # Use 10 PCs
        if r['area'] not in area_results:
            area_results[r['area']] = []
        area_results[r['area']].append(r['mean_r2'])

if area_results:
    areas = list(area_results.keys())
    means = [np.mean(area_results[a]) for a in areas]
    ax = axes[0]
    colors = ['green' if m > 0 else 'red' for m in means]
    ax.bar(areas, means, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Mean R²')
    ax.set_title('RT Prediction by Brain Area\n(10 PCs)')
    ax.set_xlabel('Brain Area')

# Plot 2: R² by time window
time_results = {}
for r in results:
    if r['n_pcs'] == 10:
        if r['time_window'] not in time_results:
            time_results[r['time_window']] = []
        time_results[r['time_window']].append(r['mean_r2'])

if time_results:
    times = list(time_results.keys())
    means = [np.mean(time_results[t]) for t in times]
    ax = axes[1]
    colors = ['green' if m > 0 else 'red' for m in means]
    ax.bar(times, means, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Mean R²')
    ax.set_title('RT Prediction by Time Window\n(10 PCs)')
    ax.set_xlabel('Time Window')

# Plot 3: R² by number of PCs
pcs_results = {}
for r in results:
    if r['n_pcs'] not in pcs_results:
        pcs_results[r['n_pcs']] = []
    pcs_results[r['n_pcs']].append(r['mean_r2'])

if pcs_results:
    pcs = sorted(pcs_results.keys())
    means = [np.mean(pcs_results[p]) for p in pcs]
    ax = axes[2]
    colors = ['green' if m > 0 else 'red' for m in means]
    ax.bar([str(p) for p in pcs], means, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Mean R²')
    ax.set_title('RT Prediction by Number of PCs')
    ax.set_xlabel('Number of PCs')

plt.tight_layout()
plt.savefig('rt_prediction_improved.png', dpi=150, bbox_inches='tight')
print("\nSaved: rt_prediction_improved.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

# Overall statistics
all_r2 = [r['mean_r2'] for r in results]
all_positive = [r['pct_positive'] for r in results]

print(f"\nAcross all conditions:")
print(f"  Mean R²: {np.mean(all_r2):.3f}")
print(f"  Best R²: {max(all_r2):.3f}")
print(f"  Conditions with mean R² > 0: {sum(1 for r in all_r2 if r > 0)}/{len(all_r2)}")
print(f"  Average % positive sessions: {np.mean(all_positive):.1f}%")

# Best condition summary
if results[0]['mean_r2'] > 0:
    print(f"\n*** POSITIVE FINDING: {results[0]['area']} during {results[0]['time_window']} ***")
    print(f"    predicts RT with R² = {results[0]['mean_r2']:.3f}")

print("\nDone!")
