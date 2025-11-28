"""
Cross-Area Communication Analysis - Robust Version
===================================================

This module extends the original cross-area analysis with rigorous controls
to address common reviewer concerns about confounding factors.

Controls Implemented:
    1. Confound Regression: Previous choice, previous reward, trial number,
       and global firing rate are included as nuisance regressors
    2. Arousal Control: Global firing rate serves as an arousal surrogate
    3. Partial Correlations: Control for global firing rate effects
    4. Cross-Mouse Validation: Leave-one-mouse-out cross-validation
    5. Sliding Time Windows: Multiple 50ms windows around stimulus onset
    6. Multiple Classifiers: L1/L2 Logistic Regression, SVM, XGBoost
    7. Electrode Quality: Correlate accuracy with recording quality metrics

Key Analyses:
    - Analysis 1: Regression with confound controls
    - Analysis 2: Cross-mouse generalization
    - Analysis 3: Partial correlations (controlling for firing rate)
    - Analysis 4: Sliding window temporal analysis
    - Analysis 5: Model comparison (multiple classifiers)
    - Analysis 6: Electrode quality effects
    - Analysis 7: Correlation direction stability

Usage:
    python cross_area_robust_analysis.py

Output:
    - Console: Detailed statistics for each analysis
    - Figure: robust_analysis_results.png (saved to ../FIGURES/)

Dependencies:
    numpy, scipy, sklearn, matplotlib, pandas
    Optional: xgboost (for XGBoost classifier comparison)

Reference:
    Steinmetz NA, Zatka-Haas P, Carandini M, Harris KD (2019).
    Distributed coding of choice, action and engagement across the mouse brain.
    Nature 576:266-273.

Author: Shir Sivroni
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Try to import xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, skipping XGBoost analysis", flush=True)

print("Loading Steinmetz data...", flush=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "DATA") + os.sep
d1 = np.load(data_path + "steinmetz_part1.npy", allow_pickle=True)['dat']
d2 = np.load(data_path + "steinmetz_part2.npy", allow_pickle=True)['dat']
d3 = np.load(data_path + "steinmetz_part3.npy", allow_pickle=True)['dat']
alldat = np.hstack([d1, d2, d3])
print(f"Loaded {len(alldat)} sessions")

# ============================================================
# Helper Functions (same as original analysis)
# ============================================================

def compute_cross_correlation(spks1, spks2, time_window=(50, 80)):
    """
    Compute cross-correlation between two areas' activity.
    spks1, spks2: (n_neurons, n_trials, n_time)
    Returns: correlation per trial
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
            if n2 >= min_neurons:
                pairs.append((area1, area2))

    return pairs

# ============================================================
# ANALYSIS 1: Add confound controls to regression
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 1: Regression with confound controls")
print("="*60)

# Collect ALL area pair data with confounds
all_correlations = []
all_choices = []
all_prev_choice = []
all_prev_reward = []
all_trial_number = []
all_global_fr = []
all_session_ids = []
all_mouse_ids = []
all_area_pairs = []

session_id = 0
for dat in alldat:
    if 'response' not in dat or dat['response'] is None:
        continue

    response = dat['response']
    valid_trials = (response == -1) | (response == 1)

    if valid_trials.sum() < 20:
        session_id += 1
        continue

    # Get all area pairs for this session
    pairs = get_area_pairs(dat, min_neurons=10)
    if len(pairs) == 0:
        session_id += 1
        continue

    choice = (response == 1).astype(int)
    n_trials = len(response)

    # Get confounds
    prev_choice = np.zeros(n_trials)
    for t in range(1, n_trials):
        prev_choice[t] = response[t-1]

    if 'prev_reward' in dat and dat['prev_reward'] is not None:
        prev_reward = dat['prev_reward'].flatten()
        if len(prev_reward) != n_trials:
            prev_reward = np.zeros(n_trials)
            if 'feedback_type' in dat:
                for t in range(1, n_trials):
                    prev_reward[t] = dat['feedback_type'][t-1]
    else:
        prev_reward = np.zeros(n_trials)
        if 'feedback_type' in dat:
            for t in range(1, n_trials):
                prev_reward[t] = dat['feedback_type'][t-1]

    trial_number = np.arange(n_trials) / n_trials

    # Global firing rate as arousal surrogate
    spks = dat['spks']
    global_fr = spks[:, :, 50:80].mean(axis=(0, 2))

    # Collect data from all pairs
    for area1, area2 in pairs:
        idx1 = dat['brain_area'] == area1
        idx2 = dat['brain_area'] == area2

        spks1 = dat['spks'][idx1]
        spks2 = dat['spks'][idx2]

        corr = compute_cross_correlation(spks1, spks2, time_window=(50, 80))

        # Store valid trials
        all_correlations.extend(corr[valid_trials])
        all_choices.extend(choice[valid_trials])
        all_prev_choice.extend(prev_choice[valid_trials])
        all_prev_reward.extend(prev_reward[valid_trials])
        all_trial_number.extend(trial_number[valid_trials])
        all_global_fr.extend(global_fr[valid_trials])
        all_session_ids.extend([session_id] * valid_trials.sum())
        all_mouse_ids.extend([dat['mouse_name']] * valid_trials.sum())
        all_area_pairs.extend([f"{area1}-{area2}"] * valid_trials.sum())

    session_id += 1

# Convert to arrays
X_corr = np.array(all_correlations).reshape(-1, 1)
X_full = np.column_stack([
    all_correlations,
    all_prev_choice,
    all_prev_reward,
    all_trial_number,
    all_global_fr
])
y = np.array(all_choices)
session_ids = np.array(all_session_ids)
mouse_ids = np.array(all_mouse_ids)

print(f"Total trial-pairs: {len(y)}")
print(f"Unique sessions: {len(np.unique(session_ids))}")
print(f"Unique mice: {len(np.unique(mouse_ids))}")

# Remove NaN values
valid_idx = ~np.isnan(X_full).any(axis=1)
X_corr = X_corr[valid_idx]
X_full = X_full[valid_idx]
y = y[valid_idx]
session_ids = session_ids[valid_idx]
mouse_ids = mouse_ids[valid_idx]

print(f"Valid trial-pairs after NaN removal: {len(y)}")

# Model 1: Correlation only (original)
scaler1 = StandardScaler()
X_corr_scaled = scaler1.fit_transform(X_corr)
model1 = LogisticRegression(random_state=42, max_iter=1000)
scores1 = cross_val_score(model1, X_corr_scaled, y, cv=5)
print(f"\nModel 1 - Correlation only:")
print(f"  Accuracy: {scores1.mean()*100:.1f}% +/- {scores1.std()*100:.1f}%")

# Model 2: All confounds (without correlation)
X_confounds = X_full[:, 1:]
scaler2 = StandardScaler()
X_confounds_scaled = scaler2.fit_transform(X_confounds)
model2 = LogisticRegression(random_state=42, max_iter=1000)
scores2 = cross_val_score(model2, X_confounds_scaled, y, cv=5)
print(f"\nModel 2 - Confounds only (prev_choice, prev_reward, trial_num, global_fr):")
print(f"  Accuracy: {scores2.mean()*100:.1f}% +/- {scores2.std()*100:.1f}%")

# Model 3: Full model
scaler3 = StandardScaler()
X_full_scaled = scaler3.fit_transform(X_full)
model3 = LogisticRegression(random_state=42, max_iter=1000)
scores3 = cross_val_score(model3, X_full_scaled, y, cv=5)
print(f"\nModel 3 - Full model (correlation + confounds):")
print(f"  Accuracy: {scores3.mean()*100:.1f}% +/- {scores3.std()*100:.1f}%")

# Fit full model to get coefficients
model3.fit(X_full_scaled, y)
feature_names = ['correlation', 'prev_choice', 'prev_reward', 'trial_number', 'global_fr']
print("\nFeature importance (coefficients):")
for name, coef in zip(feature_names, model3.coef_[0]):
    print(f"  {name}: {coef:.4f}")

# ============================================================
# ANALYSIS 2: Cross-mouse validation
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 2: Cross-mouse validation")
print("="*60)

unique_mice = np.unique(mouse_ids)
print(f"Mice: {unique_mice}")

logo = LeaveOneGroupOut()
cross_mouse_scores = []
cross_mouse_names = []

for train_idx, test_idx in logo.split(X_full_scaled, y, mouse_ids):
    if len(test_idx) < 50:  # Skip mice with too few trials
        continue
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_full_scaled[train_idx], y[train_idx])
    score = model.score(X_full_scaled[test_idx], y[test_idx])
    cross_mouse_scores.append(score)
    test_mouse = mouse_ids[test_idx[0]]
    cross_mouse_names.append(test_mouse)
    print(f"  Leave out {test_mouse}: {score*100:.1f}%")

print(f"\nCross-mouse accuracy: {np.mean(cross_mouse_scores)*100:.1f}% +/- {np.std(cross_mouse_scores)*100:.1f}%")

# ============================================================
# ANALYSIS 3: Partial correlations
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 3: Partial correlations (controlling for firing rate)")
print("="*60)

def partial_correlation(x, y, z):
    """Compute partial correlation between x and y, controlling for z"""
    from scipy.stats import pearsonr

    slope_x, intercept_x = np.polyfit(z, x, 1)
    x_resid = x - (slope_x * z + intercept_x)

    slope_y, intercept_y = np.polyfit(z, y, 1)
    y_resid = y - (slope_y * z + intercept_y)

    return pearsonr(x_resid, y_resid)

corr_array = np.array(all_correlations)[valid_idx]
fr_array = np.array(all_global_fr)[valid_idx]

r_simple, p_simple = stats.pearsonr(corr_array, y)
print(f"Simple correlation (correlation vs choice): r={r_simple:.4f}, p={p_simple:.2e}")

r_partial, p_partial = partial_correlation(corr_array, y.astype(float), fr_array)
print(f"Partial correlation (controlling for global FR): r={r_partial:.4f}, p={p_partial:.2e}")

# ============================================================
# ANALYSIS 4: Multiple time windows
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 4: Sliding window analysis")
print("="*60)

time_windows = [
    (10, 40),   # Pre-stimulus
    (25, 55),   # Around stimulus
    (40, 70),   # Early post-stimulus
    (50, 80),   # Original window
    (60, 90),   # Late post-stimulus
    (70, 100),  # Later
]

window_results = []

for tw in time_windows:
    tw_correlations = []
    tw_choices = []

    for dat in alldat:
        if 'response' not in dat or dat['response'] is None:
            continue

        response = dat['response']
        valid_trials = (response == -1) | (response == 1)

        if valid_trials.sum() < 20:
            continue

        pairs = get_area_pairs(dat, min_neurons=10)
        if len(pairs) == 0:
            continue

        choice = (response == 1).astype(int)

        for area1, area2 in pairs:
            idx1 = dat['brain_area'] == area1
            idx2 = dat['brain_area'] == area2

            spks1 = dat['spks'][idx1]
            spks2 = dat['spks'][idx2]

            if spks1.shape[2] <= tw[1] or spks2.shape[2] <= tw[1]:
                continue

            corr = compute_cross_correlation(spks1, spks2, time_window=tw)
            tw_correlations.extend(corr[valid_trials])
            tw_choices.extend(choice[valid_trials])

    if len(tw_correlations) > 100:
        X_tw = np.array(tw_correlations).reshape(-1, 1)
        y_tw = np.array(tw_choices)

        valid_tw = ~np.isnan(X_tw).flatten()
        X_tw = X_tw[valid_tw]
        y_tw = y_tw[valid_tw]

        scaler_tw = StandardScaler()
        X_tw_scaled = scaler_tw.fit_transform(X_tw)

        model_tw = LogisticRegression(random_state=42, max_iter=1000)
        scores_tw = cross_val_score(model_tw, X_tw_scaled, y_tw, cv=5)

        window_results.append({
            'window': tw,
            'accuracy': scores_tw.mean(),
            'std': scores_tw.std(),
            'n_trials': len(y_tw)
        })

        label = f"Window {tw[0]*10-250:+4d} to {tw[1]*10-250:+4d}ms"
        print(f"{label}: {scores_tw.mean()*100:.1f}% +/- {scores_tw.std()*100:.1f}% (n={len(y_tw)})")

# ============================================================
# ANALYSIS 5: Multiple models comparison
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 5: Multiple models comparison")
print("="*60)

# Subsample for SVM (too slow on 265K samples)
max_samples = 20000
if len(y) > max_samples:
    np.random.seed(42)
    subsample_idx = np.random.choice(len(y), max_samples, replace=False)
    X_sub = X_full_scaled[subsample_idx]
    y_sub = y[subsample_idx]
    print(f"Subsampled to {max_samples} for SVM models")
else:
    X_sub = X_full_scaled
    y_sub = y

models = {
    'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), X_full_scaled, y),
    'L1 Logistic (Lasso)': (LogisticRegressionCV(penalty='l1', solver='saga', cv=5, random_state=42, max_iter=1000), X_full_scaled, y),
    'Linear SVM': (SVC(kernel='linear', random_state=42), X_sub, y_sub),
    'RBF SVM': (SVC(kernel='rbf', random_state=42), X_sub, y_sub),
}

if HAS_XGBOOST:
    models['XGBoost'] = (XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0), X_full_scaled, y)

model_results = {}
for name, (model, X_m, y_m) in models.items():
    scores = cross_val_score(model, X_m, y_m, cv=5)
    model_results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name}: {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%", flush=True)

# ============================================================
# ANALYSIS 6: Electrode quality correlation
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 6: Per-session accuracy vs electrode quality")
print("="*60)

# Compute accuracy per session
session_accuracies = defaultdict(list)
session_info = {}

for i, (corr_val, choice_val, sess_id) in enumerate(zip(
    np.array(all_correlations)[valid_idx], y, session_ids)):
    session_accuracies[sess_id].append((corr_val, choice_val))

session_acc_list = []
session_n_units_list = []
session_mean_fr_list = []

for sess_id in np.unique(session_ids):
    data = session_accuracies[sess_id]
    if len(data) < 50:
        continue

    corrs = np.array([d[0] for d in data]).reshape(-1, 1)
    choices = np.array([d[1] for d in data])

    if np.isnan(corrs).any():
        continue

    scaler = StandardScaler()
    corrs_scaled = scaler.fit_transform(corrs)

    model = LogisticRegression(random_state=42, max_iter=1000)
    try:
        scores = cross_val_score(model, corrs_scaled, choices, cv=min(5, len(choices)//10))
        acc = scores.mean()

        dat = alldat[int(sess_id)]
        n_units = dat['spks'].shape[0]
        mean_fr = dat['spks'].mean()

        session_acc_list.append(acc)
        session_n_units_list.append(n_units)
        session_mean_fr_list.append(mean_fr)
    except:
        continue

if len(session_acc_list) > 5:
    r_units, p_units = stats.pearsonr(session_acc_list, session_n_units_list)
    r_fr, p_fr = stats.pearsonr(session_acc_list, session_mean_fr_list)

    print(f"Correlation with number of units: r={r_units:.3f}, p={p_units:.3f}")
    print(f"Correlation with mean firing rate: r={r_fr:.3f}, p={p_fr:.3f}")
    print(f"Tested {len(session_acc_list)} sessions")
else:
    r_units, p_units = 0, 1
    r_fr, p_fr = 0, 1
    print("Not enough sessions for electrode quality analysis")

# ============================================================
# ANALYSIS 7: Correlation direction stability
# ============================================================
print("\n" + "="*60)
print("ANALYSIS 7: Correlation direction stability across sessions")
print("="*60)

pair_directions = defaultdict(list)

for dat in alldat:
    if 'response' not in dat or dat['response'] is None:
        continue

    response = dat['response']
    valid_trials = (response == -1) | (response == 1)

    if valid_trials.sum() < 20:
        continue

    pairs = get_area_pairs(dat, min_neurons=10)
    choice = (response == 1).astype(int)

    for area1, area2 in pairs:
        idx1 = dat['brain_area'] == area1
        idx2 = dat['brain_area'] == area2

        spks1 = dat['spks'][idx1]
        spks2 = dat['spks'][idx2]

        corr = compute_cross_correlation(spks1, spks2, time_window=(50, 80))
        corr_valid = corr[valid_trials]
        choice_valid = choice[valid_trials]

        mean_corr_left = corr_valid[choice_valid == 0].mean()
        mean_corr_right = corr_valid[choice_valid == 1].mean()

        direction = 1 if mean_corr_right > mean_corr_left else -1
        pair_directions[f"{area1}-{area2}"].append(direction)

print("Consistency of correlation direction across sessions (pairs with >= 3 sessions):")
stable_pairs = 0
total_pairs = 0
for pair, directions in sorted(pair_directions.items(), key=lambda x: -len(x[1])):
    if len(directions) >= 3:
        total_pairs += 1
        consistency = abs(np.mean(directions))
        n_positive = sum(d > 0 for d in directions)
        n_negative = sum(d < 0 for d in directions)
        if consistency >= 0.6:  # Mostly consistent
            stable_pairs += 1
        print(f"  {pair}: {n_positive}+/{n_negative}- (consistency: {consistency:.2f})")

if total_pairs > 0:
    print(f"\n{stable_pairs}/{total_pairs} pairs ({100*stable_pairs/total_pairs:.0f}%) show consistent direction")

# ============================================================
# SUMMARY FIGURE
# ============================================================
print("\n" + "="*60)
print("Creating summary figure...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Model comparison with confounds
ax1 = axes[0, 0]
models_names = ['Corr only', 'Confounds\nonly', 'Full model']
accuracies = [scores1.mean()*100, scores2.mean()*100, scores3.mean()*100]
errors = [scores1.std()*100, scores2.std()*100, scores3.std()*100]
bars = ax1.bar(models_names, accuracies, yerr=errors, capsize=5, color=['#3498db', '#e74c3c', '#2ecc71'])
ax1.axhline(y=50, color='gray', linestyle='--', label='Chance')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('A. Confound Control Analysis')
ax1.set_ylim(45, 65)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{acc:.1f}%',
             ha='center', va='bottom', fontsize=10)

# 2. Cross-mouse validation
ax2 = axes[0, 1]
if len(cross_mouse_scores) > 0:
    mice_names = [f'{m[:4]}' for m in cross_mouse_names]
    ax2.bar(mice_names, [s*100 for s in cross_mouse_scores], color='#9b59b6')
    ax2.axhline(y=50, color='gray', linestyle='--', label='Chance')
    ax2.axhline(y=np.mean(cross_mouse_scores)*100, color='red', linestyle='-',
                label=f'Mean: {np.mean(cross_mouse_scores)*100:.1f}%')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('B. Cross-Mouse Validation')
    ax2.set_ylim(45, 65)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.tick_params(axis='x', rotation=45)

# 3. Time window analysis
ax3 = axes[0, 2]
if window_results:
    windows = [f"{(r['window'][0]*10-250)}" for r in window_results]
    accs = [r['accuracy']*100 for r in window_results]
    stds = [r['std']*100 for r in window_results]

    x_pos = range(len(windows))
    ax3.errorbar(x_pos, accs, yerr=stds, marker='o', capsize=5, color='#1abc9c', linewidth=2, markersize=8)
    ax3.axhline(y=50, color='gray', linestyle='--')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(windows, rotation=45)
    ax3.set_xlabel('Window start (ms from stim)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('C. Sliding Window Analysis')

# 4. Multiple models
ax4 = axes[1, 0]
model_names_list = list(model_results.keys())
model_accs = [model_results[m]['mean']*100 for m in model_names_list]
model_stds = [model_results[m]['std']*100 for m in model_names_list]
bars = ax4.barh(model_names_list, model_accs, xerr=model_stds, capsize=5, color='#f39c12')
ax4.axvline(x=50, color='gray', linestyle='--')
ax4.set_xlabel('Accuracy (%)')
ax4.set_title('D. Model Comparison')
ax4.set_xlim(45, 65)

# 5. Feature importance
ax5 = axes[1, 1]
coefs = model3.coef_[0]
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coefs]
bars = ax5.barh(feature_names, np.abs(coefs), color=colors)
ax5.set_xlabel('|Coefficient|')
ax5.set_title('E. Feature Importance (Full Model)')
ax5.axvline(x=0, color='gray', linestyle='-')

# 6. Summary text
ax6 = axes[1, 2]
ax6.axis('off')

# Calculate key statistics
improvement = scores3.mean()*100 - scores2.mean()*100

summary_text = f"""ROBUST ANALYSIS SUMMARY

Original finding:
  Cross-area correlation predicts choice
  Accuracy: {scores1.mean()*100:.1f}%

After controlling for confounds:
  Confounds alone: {scores2.mean()*100:.1f}%
  Full model: {scores3.mean()*100:.1f}%
  Improvement: {improvement:+.1f}%

Cross-mouse generalization:
  {np.mean(cross_mouse_scores)*100:.1f}% +/- {np.std(cross_mouse_scores)*100:.1f}%

Partial correlation (controlling for FR):
  r = {r_partial:.4f}, p = {p_partial:.2e}

Electrode quality effect:
  r(accuracy, n_units) = {r_units:.3f}
"""
ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig_path = os.path.join(script_dir, '..', 'FIGURES', 'robust_analysis_results.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nFigure saved to {fig_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY - ADDRESSING REVIEWER CONCERNS")
print("="*60)

print(f"""
1. STATE-DRIVEN vs DECISION-RELATED:
   - Confounds alone predict: {scores2.mean()*100:.1f}%
   - Full model (with correlation): {scores3.mean()*100:.1f}%
   - Correlation adds: {improvement:+.1f}% beyond confounds
   - Partial correlation (controlling arousal): r={r_partial:.4f}, p={p_partial:.2e}

2. ACCURACY LEVEL:
   - Correlation only: {scores1.mean()*100:.1f}%
   - Best model: {max(model_results.values(), key=lambda x: x['mean'])['mean']*100:.1f}%

3. CROSS-MOUSE GENERALIZATION:
   - Mean accuracy: {np.mean(cross_mouse_scores)*100:.1f}% +/- {np.std(cross_mouse_scores)*100:.1f}%
   - Tested on {len(cross_mouse_scores)} mice

4. ELECTRODE QUALITY:
   - Correlation with n_units: r={r_units:.3f}, p={p_units:.3f}
   - Effect is {'not ' if p_units > 0.05 else ''}related to recording quality
""")

print("\nAnalysis complete!")
