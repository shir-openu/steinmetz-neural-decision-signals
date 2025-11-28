"""
Pre-stimulus Inter-area vs Post-stimulus Local Activity Analysis
================================================================

New research question based on literature:
- Pre-stimulus: inter-area correlations predict choice (bias signal)
- Post-stimulus: local activity predicts choice (evidence signal)

This is a more nuanced and literature-supported hypothesis.

References:
- Arieli et al., 1996; Hesselmann et al., 2008 (pre-stimulus states)
- Fries 2005/2015; Donner et al., 2009 (inter-areal interactions)
- Steinmetz et al., 2019; Mante et al., 2013 (post-stimulus local coding)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)

print("="*70)
print("PRE-STIMULUS INTER-AREA vs POST-STIMULUS LOCAL ANALYSIS")
print("="*70)

# Load data
print("\nLoading Steinmetz data...", flush=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "DATA") + os.sep
d1 = np.load(data_path + "steinmetz_part1.npy", allow_pickle=True)['dat']
d2 = np.load(data_path + "steinmetz_part2.npy", allow_pickle=True)['dat']
d3 = np.load(data_path + "steinmetz_part3.npy", allow_pickle=True)['dat']
alldat = np.hstack([d1, d2, d3])
print(f"Loaded {len(alldat)} sessions", flush=True)

# Time windows (in 10ms bins, stimulus onset at bin 50 = 0ms)
# Pre-stimulus: -500ms to 0ms (bins 0-50)
# Post-stimulus: 0ms to +500ms (bins 50-100)
TIME_WINDOWS = {
    'pre_early': (0, 25),      # -500 to -250ms
    'pre_late': (25, 50),      # -250 to 0ms
    'post_early': (50, 75),    # 0 to +250ms
    'post_late': (75, 100),    # +250 to +500ms
}

MIN_NEURONS = 10

def compute_cross_correlation(spks1, spks2, time_window):
    """Compute trial-by-trial cross-area correlation."""
    t_start, t_end = time_window

    # Average over neurons to get population activity
    pop1 = spks1[:, :, t_start:t_end].mean(axis=0)  # (n_trials, n_time)
    pop2 = spks2[:, :, t_start:t_end].mean(axis=0)

    n_trials = pop1.shape[0]
    corrs = np.zeros(n_trials)

    for trial in range(n_trials):
        if pop1[trial].std() > 0 and pop2[trial].std() > 0:
            corrs[trial] = np.corrcoef(pop1[trial], pop2[trial])[0, 1]

    return corrs


def compute_local_activity(spks, time_window):
    """Compute mean firing rate per trial (local activity)."""
    t_start, t_end = time_window
    return spks[:, :, t_start:t_end].mean(axis=(0, 2))  # (n_trials,)


def get_area_pairs(session, min_neurons=MIN_NEURONS):
    """Get all pairs of areas with enough neurons."""
    areas = np.unique(session['brain_area'])
    pairs = []

    for i, area1 in enumerate(areas):
        n1 = np.sum(session['brain_area'] == area1)
        if n1 < min_neurons:
            continue
        for area2 in areas[i+1:]:
            n2 = np.sum(session['brain_area'] == area2)
            if n2 >= min_neurons:
                pairs.append((area1, area2))

    return pairs


def decode_choice(X, y, cv=5):
    """Decode choice using logistic regression with cross-validation."""
    if len(np.unique(y)) < 2 or len(y) < 20:
        return np.nan, np.nan

    # Remove NaN
    valid = ~np.isnan(X).any(axis=1) if X.ndim > 1 else ~np.isnan(X)
    X = X[valid] if X.ndim > 1 else X[valid].reshape(-1, 1)
    y = y[valid]

    if len(y) < 20 or len(np.unique(y)) < 2:
        return np.nan, np.nan

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    try:
        scores = cross_val_score(model, X_scaled, y, cv=min(cv, len(y)//10), scoring='accuracy')
        return scores.mean(), scores.std()
    except:
        return np.nan, np.nan


# ============================================================
# MAIN ANALYSIS
# ============================================================

print("\n" + "="*70)
print("ANALYSIS: Comparing inter-area vs local activity across time")
print("="*70)

results = {tw: {'inter_area': [], 'local': []} for tw in TIME_WINDOWS.keys()}

for session_idx, dat in enumerate(alldat):
    if 'response' not in dat or dat['response'] is None:
        continue

    response = dat['response']
    valid_trials = (response == -1) | (response == 1)

    if valid_trials.sum() < 30:
        continue

    choice = (response[valid_trials] == 1).astype(int)
    spks = dat['spks']
    brain_areas = dat['brain_area']

    # Check time dimension
    n_time = spks.shape[2]
    if n_time < 100:
        continue

    pairs = get_area_pairs(dat, min_neurons=MIN_NEURONS)
    if len(pairs) == 0:
        continue

    # Get areas with enough neurons
    unique_areas = np.unique(brain_areas)
    good_areas = [a for a in unique_areas if np.sum(brain_areas == a) >= MIN_NEURONS]

    for tw_name, tw in TIME_WINDOWS.items():
        # Skip if time window exceeds data
        if tw[1] > n_time:
            continue

        # 1. Inter-area correlations
        inter_area_features = []
        for area1, area2 in pairs[:10]:  # Limit pairs for speed
            idx1 = brain_areas == area1
            idx2 = brain_areas == area2

            spks1 = spks[idx1]
            spks2 = spks[idx2]

            corr = compute_cross_correlation(spks1, spks2, tw)
            inter_area_features.append(corr[valid_trials])

        if len(inter_area_features) > 0:
            X_inter = np.column_stack(inter_area_features)
            acc_inter, _ = decode_choice(X_inter, choice)
            if not np.isnan(acc_inter):
                results[tw_name]['inter_area'].append(acc_inter)

        # 2. Local activity (firing rates per area)
        local_features = []
        for area in good_areas[:10]:  # Limit areas for speed
            idx = brain_areas == area
            area_spks = spks[idx]

            local_fr = compute_local_activity(area_spks, tw)
            local_features.append(local_fr[valid_trials])

        if len(local_features) > 0:
            X_local = np.column_stack(local_features)
            acc_local, _ = decode_choice(X_local, choice)
            if not np.isnan(acc_local):
                results[tw_name]['local'].append(acc_local)

print(f"\nProcessed {session_idx + 1} sessions", flush=True)

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

summary_data = {}

print("\n{:<15} {:>20} {:>20}".format("Time Window", "Inter-area Corr", "Local Activity"))
print("-"*60)

for tw_name in ['pre_early', 'pre_late', 'post_early', 'post_late']:
    inter_acc = results[tw_name]['inter_area']
    local_acc = results[tw_name]['local']

    inter_mean = np.mean(inter_acc) if len(inter_acc) > 0 else np.nan
    inter_std = np.std(inter_acc) if len(inter_acc) > 0 else np.nan
    local_mean = np.mean(local_acc) if len(local_acc) > 0 else np.nan
    local_std = np.std(local_acc) if len(local_acc) > 0 else np.nan

    summary_data[tw_name] = {
        'inter_mean': inter_mean, 'inter_std': inter_std,
        'local_mean': local_mean, 'local_std': local_std,
        'inter_n': len(inter_acc), 'local_n': len(local_acc)
    }

    tw_label = tw_name.replace('_', ' ').title()
    print(f"{tw_label:<15} {inter_mean*100:>8.1f}% +/- {inter_std*100:>4.1f}%    {local_mean*100:>8.1f}% +/- {local_std*100:>4.1f}%")

# ============================================================
# STATISTICAL TESTS
# ============================================================

print("\n" + "="*70)
print("STATISTICAL TESTS")
print("="*70)

# Test 1: Pre-stimulus inter-area vs chance
pre_inter = results['pre_late']['inter_area']
if len(pre_inter) > 5:
    t_stat, p_val = stats.ttest_1samp(pre_inter, 0.5)
    print(f"\n1. Pre-stimulus inter-area vs chance (0.5):")
    print(f"   t={t_stat:.3f}, p={p_val:.4f}")
    print(f"   {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'}")

# Test 2: Post-stimulus local vs chance
post_local = results['post_early']['local']
if len(post_local) > 5:
    t_stat, p_val = stats.ttest_1samp(post_local, 0.5)
    print(f"\n2. Post-stimulus local vs chance (0.5):")
    print(f"   t={t_stat:.3f}, p={p_val:.4f}")
    print(f"   {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'}")

# Test 3: Pre-stimulus: inter-area vs local
pre_inter = results['pre_late']['inter_area']
pre_local = results['pre_late']['local']
if len(pre_inter) > 5 and len(pre_local) > 5:
    # Use Mann-Whitney since samples may differ
    u_stat, p_val = stats.mannwhitneyu(pre_inter, pre_local, alternative='greater')
    print(f"\n3. Pre-stimulus: inter-area > local?")
    print(f"   U={u_stat:.1f}, p={p_val:.4f}")
    print(f"   {'YES - inter-area dominates pre-stimulus' if p_val < 0.05 else 'No significant difference'}")

# Test 4: Post-stimulus: local vs inter-area
post_inter = results['post_early']['inter_area']
post_local = results['post_early']['local']
if len(post_inter) > 5 and len(post_local) > 5:
    u_stat, p_val = stats.mannwhitneyu(post_local, post_inter, alternative='greater')
    print(f"\n4. Post-stimulus: local > inter-area?")
    print(f"   U={u_stat:.1f}, p={p_val:.4f}")
    print(f"   {'YES - local dominates post-stimulus' if p_val < 0.05 else 'No significant difference'}")

# ============================================================
# VISUALIZATION
# ============================================================

print("\n" + "="*70)
print("Creating figure...")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Accuracy over time
ax1 = axes[0]
time_labels = ['Pre\n(-500 to -250ms)', 'Pre\n(-250 to 0ms)',
               'Post\n(0 to +250ms)', 'Post\n(+250 to +500ms)']
x_pos = np.arange(4)

inter_means = [summary_data[tw]['inter_mean']*100 for tw in ['pre_early', 'pre_late', 'post_early', 'post_late']]
inter_stds = [summary_data[tw]['inter_std']*100 for tw in ['pre_early', 'pre_late', 'post_early', 'post_late']]
local_means = [summary_data[tw]['local_mean']*100 for tw in ['pre_early', 'pre_late', 'post_early', 'post_late']]
local_stds = [summary_data[tw]['local_std']*100 for tw in ['pre_early', 'pre_late', 'post_early', 'post_late']]

width = 0.35
ax1.bar(x_pos - width/2, inter_means, width, yerr=inter_stds, label='Inter-area Correlation',
        color='#3498db', capsize=5, alpha=0.8)
ax1.bar(x_pos + width/2, local_means, width, yerr=local_stds, label='Local Activity',
        color='#e74c3c', capsize=5, alpha=0.8)
ax1.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='Chance')
ax1.axvline(x=1.5, color='black', linestyle=':', linewidth=2, alpha=0.5)
ax1.text(1.5, ax1.get_ylim()[1]*0.95, 'Stimulus\nOnset', ha='center', fontsize=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(time_labels)
ax1.set_ylabel('Decoding Accuracy (%)')
ax1.set_title('A. Choice Decoding Across Time')
ax1.legend(loc='upper left')
ax1.set_ylim(45, 70)

# Plot 2: Pre-stimulus comparison
ax2 = axes[1]
pre_inter_all = results['pre_late']['inter_area']
pre_local_all = results['pre_late']['local']

bp = ax2.boxplot([pre_inter_all, pre_local_all], labels=['Inter-area\nCorrelation', 'Local\nActivity'])
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=2)
ax2.set_ylabel('Accuracy')
ax2.set_title('B. Pre-stimulus Period\n(-250 to 0ms)')

# Add significance annotation
if len(pre_inter_all) > 5 and len(pre_local_all) > 5:
    u_stat, p_val = stats.mannwhitneyu(pre_inter_all, pre_local_all)
    sig_text = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
    y_max = max(max(pre_inter_all), max(pre_local_all))
    ax2.plot([1, 2], [y_max + 0.02, y_max + 0.02], 'k-', linewidth=1.5)
    ax2.text(1.5, y_max + 0.03, sig_text, ha='center', fontsize=10)

# Plot 3: Post-stimulus comparison
ax3 = axes[2]
post_inter_all = results['post_early']['inter_area']
post_local_all = results['post_early']['local']

bp = ax3.boxplot([post_inter_all, post_local_all], labels=['Inter-area\nCorrelation', 'Local\nActivity'])
ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=2)
ax3.set_ylabel('Accuracy')
ax3.set_title('C. Post-stimulus Period\n(0 to +250ms)')

# Add significance annotation
if len(post_inter_all) > 5 and len(post_local_all) > 5:
    u_stat, p_val = stats.mannwhitneyu(post_local_all, post_inter_all)
    sig_text = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'
    y_max = max(max(post_inter_all), max(post_local_all))
    ax3.plot([1, 2], [y_max + 0.02, y_max + 0.02], 'k-', linewidth=1.5)
    ax3.text(1.5, y_max + 0.03, sig_text, ha='center', fontsize=10)

plt.tight_layout()
fig_path = os.path.join(script_dir, '..', 'FIGURES', 'pre_vs_post_analysis.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Figure saved to: {fig_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("FINAL SUMMARY - NEW RESEARCH QUESTION")
print("="*70)

print("""
HYPOTHESIS (based on literature):
- Pre-stimulus: Inter-area correlations should dominate (bias signal)
- Post-stimulus: Local activity should dominate (evidence signal)

RESULTS:
""")

pre_inter_mean = summary_data['pre_late']['inter_mean']
pre_local_mean = summary_data['pre_late']['local_mean']
post_inter_mean = summary_data['post_early']['inter_mean']
post_local_mean = summary_data['post_early']['local_mean']

print(f"PRE-STIMULUS (-250 to 0ms):")
print(f"  Inter-area: {pre_inter_mean*100:.1f}%")
print(f"  Local:      {pre_local_mean*100:.1f}%")
print(f"  Difference: {(pre_inter_mean - pre_local_mean)*100:+.1f}%")

print(f"\nPOST-STIMULUS (0 to +250ms):")
print(f"  Inter-area: {post_inter_mean*100:.1f}%")
print(f"  Local:      {post_local_mean*100:.1f}%")
print(f"  Difference: {(post_local_mean - post_inter_mean)*100:+.1f}%")

# Determine if hypothesis is supported
pre_inter_wins = pre_inter_mean > pre_local_mean
post_local_wins = post_local_mean > post_inter_mean

if pre_inter_wins and post_local_wins:
    print("\n*** HYPOTHESIS SUPPORTED ***")
    print("Inter-area correlations dominate pre-stimulus")
    print("Local activity dominates post-stimulus")
    print("\nThis supports the dual-mechanism model of perceptual choice!")
elif not pre_inter_wins and not post_local_wins:
    print("\n*** HYPOTHESIS NOT SUPPORTED ***")
    print("Results are opposite to prediction")
else:
    print("\n*** PARTIAL SUPPORT ***")
    if pre_inter_wins:
        print("Pre-stimulus: Inter-area wins (as predicted)")
    else:
        print("Pre-stimulus: Local wins (opposite to prediction)")
    if post_local_wins:
        print("Post-stimulus: Local wins (as predicted)")
    else:
        print("Post-stimulus: Inter-area wins (opposite to prediction)")

print("\nAnalysis complete!")
