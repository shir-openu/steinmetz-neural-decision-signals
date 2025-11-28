"""
Reaction Time Prediction from Neural Activity
==============================================
Can we predict how fast the mouse will respond based on neural activity?

Hypothesis: Pre-stimulus or early-stimulus neural activity contains information
about the upcoming reaction time (related to attention, arousal, or decision confidence).
"""

import numpy as np
import os
from sklearn.linear_model import Ridge
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
# PART 1: Explore RT data
# =============================================================================

print("\n" + "="*60)
print("PART 1: Exploring Reaction Time Data")
print("="*60)

# Check what RT-related fields exist
session = alldat[11]  # Example session
print(f"\nChecking session {11}: {session['mouse_name']}")

# The dataset has reaction_time field
if 'reaction_time' in session:
    rt = session['reaction_time']
    print(f"reaction_time shape: {rt.shape}")
    print(f"First 10 RTs (col 0): {rt[:10, 0]}")
    print(f"First 10 RTs (col 1): {rt[:10, 1]}")

    # RT is already in ms in column 0
    rts = rt[:, 0] if rt.ndim > 1 else rt
    # Filter valid: positive, finite, and reasonable range
    valid_rts = rts[(rts > 0) & (rts < 2000) & np.isfinite(rts)]
    print(f"Valid RTs (0 < rt < 2000ms): {len(valid_rts)}/{len(rts)}")
    if len(valid_rts) > 0:
        print(f"RT range: {valid_rts.min():.0f}ms - {valid_rts.max():.0f}ms")
        print(f"RT mean: {valid_rts.mean():.0f}ms, std: {valid_rts.std():.0f}ms")

# =============================================================================
# PART 2: Build RT Prediction Model
# =============================================================================

print("\n" + "="*60)
print("PART 2: Building RT Prediction Model")
print("="*60)

def prepare_rt_prediction_data(session, area='VISp', time_window='pre_stim'):
    """
    Prepare features (neural activity) and target (RT) for prediction.

    time_window:
        'pre_stim' - activity before stimulus (0-500ms)
        'early_stim' - early stimulus period (500-600ms)
        'late_stim' - later stimulus period (600-800ms)
    """
    # Check if session has reaction_time
    if 'reaction_time' not in session:
        return None, None, None

    # Get neurons from specified area
    area_mask = session['brain_area'] == area
    if np.sum(area_mask) < 10:
        return None, None, None

    spks = session['spks'][area_mask]
    n_neurons = spks.shape[0]
    n_trials = spks.shape[1]

    # Get reaction time (already in ms)
    rt = session['reaction_time']
    rts = rt[:, 0] if rt.ndim > 1 else rt
    # Already in ms, no conversion needed

    # Define time windows (in 10ms bins)
    if time_window == 'pre_stim':
        t_start, t_end = 30, 50  # 300-500ms (before stim at 500ms)
    elif time_window == 'early_stim':
        t_start, t_end = 50, 60  # 500-600ms (first 100ms after stim)
    elif time_window == 'late_stim':
        t_start, t_end = 60, 80  # 600-800ms
    else:
        raise ValueError(f"Unknown time window: {time_window}")

    # Extract features: mean firing rate in window
    X = spks[:, :, t_start:t_end].mean(axis=2).T  # (n_trials, n_neurons)
    y = rts

    # Filter valid trials (non-nan RT, response trials only)
    responses = session['response']
    valid = np.isfinite(y) & (responses != 0)  # Only left/right responses

    # Also filter by reasonable RT range (50-1500ms)
    valid = valid & (y > 50) & (y < 1500)

    X = X[valid]
    y = y[valid]

    return X, y, n_neurons

# Test on multiple sessions
results = []

areas_to_test = ['VISp', 'MOs', 'MOp', 'ACA', 'SC']
time_windows = ['pre_stim', 'early_stim', 'late_stim']

print("\nTesting RT prediction across sessions and areas...")
print(f"{'Area':<8} {'Window':<12} {'N_sess':<8} {'R²':<10} {'p-value':<10}")
print("-" * 50)

for area in areas_to_test:
    for time_window in time_windows:
        all_r2 = []

        for session_idx, session in enumerate(alldat):
            X, y, n_neurons = prepare_rt_prediction_data(session, area, time_window)

            if X is None or len(X) < 50:
                continue

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Ridge regression with cross-validation
            model = Ridge(alpha=1.0)
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            mean_r2 = np.mean(scores)

            if mean_r2 > -1:  # Filter out very bad fits
                all_r2.append(mean_r2)
                results.append({
                    'area': area,
                    'time_window': time_window,
                    'session': session_idx,
                    'r2': mean_r2,
                    'n_neurons': n_neurons,
                    'n_trials': len(y)
                })

        if len(all_r2) > 0:
            mean_r2 = np.mean(all_r2)
            # Test if significantly > 0
            if len(all_r2) > 1:
                t_stat, p_val = stats.ttest_1samp(all_r2, 0)
            else:
                p_val = 1.0
            print(f"{area:<8} {time_window:<12} {len(all_r2):<8} {mean_r2:<10.3f} {p_val:<10.4f}")

# =============================================================================
# PART 4: Detailed Analysis of Best Conditions
# =============================================================================

print("\n" + "="*60)
print("PART 4: Detailed Analysis")
print("="*60)

if len(results) > 0:
    # Find best area/time_window combination
    import pandas as pd
    df = pd.DataFrame(results)

    # Group by area and time_window
    grouped = df.groupby(['area', 'time_window'])['r2'].agg(['mean', 'std', 'count'])
    grouped = grouped.reset_index()
    grouped = grouped.sort_values('mean', ascending=False)

    print("\nBest combinations (sorted by mean R²):")
    print(grouped.head(10).to_string())

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot by area
    area_means = df.groupby('area')['r2'].mean()
    area_stds = df.groupby('area')['r2'].std()

    ax = axes[0]
    bars = ax.bar(area_means.index, area_means.values, yerr=area_stds.values,
                  capsize=5, color='steelblue', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('R² (cross-validated)')
    ax.set_title('RT Prediction by Brain Area')
    ax.set_xlabel('Brain Area')

    # Bar plot by time window
    time_means = df.groupby('time_window')['r2'].mean()
    time_stds = df.groupby('time_window')['r2'].std()

    ax = axes[1]
    bars = ax.bar(time_means.index, time_means.values, yerr=time_stds.values,
                  capsize=5, color='forestgreen', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('R² (cross-validated)')
    ax.set_title('RT Prediction by Time Window')
    ax.set_xlabel('Time Window')

    plt.tight_layout()
    plt.savefig('rt_prediction_results.png', dpi=150, bbox_inches='tight')
    print("\nSaved: rt_prediction_results.png")

    # =============================================================================
    # PART 5: Best Session Deep Dive
    # =============================================================================

    print("\n" + "="*60)
    print("PART 5: Best Session Analysis")
    print("="*60)

    # Find best session
    best_idx = df['r2'].idxmax()
    best = df.loc[best_idx]
    print(f"\nBest result: Area={best['area']}, Window={best['time_window']}, "
          f"Session={best['session']}, R²={best['r2']:.3f}")

    # Detailed analysis of best session
    session = alldat[int(best['session'])]
    X, y, n_neurons = prepare_rt_prediction_data(session, best['area'], best['time_window'])

    if X is not None:
        # Fit model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        # Plot actual vs predicted
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

        # Scatter plot
        ax = axes2[0]
        ax.scatter(y, y_pred, alpha=0.5, s=20)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
        ax.set_xlabel('Actual RT (ms)')
        ax.set_ylabel('Predicted RT (ms)')
        ax.set_title(f'Actual vs Predicted RT\nR² = {best["r2"]:.3f}')

        # RT distribution
        ax = axes2[1]
        ax.hist(y, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Reaction Time (ms)')
        ax.set_ylabel('Count')
        ax.set_title('RT Distribution')
        ax.axvline(np.median(y), color='red', linestyle='--', label=f'Median={np.median(y):.0f}ms')
        ax.legend()

        # Feature importance (top neurons)
        ax = axes2[2]
        coef_abs = np.abs(model.coef_)
        top_idx = np.argsort(coef_abs)[-10:]
        ax.barh(range(10), coef_abs[top_idx], color='forestgreen', alpha=0.7)
        ax.set_xlabel('|Coefficient|')
        ax.set_ylabel('Neuron Rank')
        ax.set_title('Top 10 Predictive Neurons')

        plt.tight_layout()
        plt.savefig('rt_prediction_best_session.png', dpi=150, bbox_inches='tight')
        print("Saved: rt_prediction_best_session.png")

else:
    print("No valid results to analyze")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if len(results) > 0:
    df = pd.DataFrame(results)
    overall_mean = df['r2'].mean()
    overall_std = df['r2'].std()
    positive_pct = (df['r2'] > 0).mean() * 100

    print(f"\nOverall R²: {overall_mean:.3f} ± {overall_std:.3f}")
    print(f"Sessions with R² > 0: {positive_pct:.0f}%")

    # Statistical test
    t_stat, p_val = stats.ttest_1samp(df['r2'].values, 0)
    print(f"\nOne-sample t-test (R² > 0): t={t_stat:.2f}, p={p_val:.4f}")

    if p_val < 0.05 and overall_mean > 0:
        print("\n*** POSITIVE RESULT: Neural activity predicts reaction time! ***")
    else:
        print("\nResult: No significant prediction of RT from neural activity")

print("\nDone!")
