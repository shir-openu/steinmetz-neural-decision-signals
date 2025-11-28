"""
Direction Field Analysis - Higher PCs (PC3-PC4)
Maybe the decision is encoded in higher principal components?
"""

import numpy as np
import os
from sklearn.decomposition import PCA
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

# Use session with most MOs neurons (session 30)
session = alldat[30]
print(f"Mouse: {session['mouse_name']}, Date: {session['date_exp']}")

area_mask = session['brain_area'] == 'MOs'
spks = session['spks'][area_mask]
responses = session['response']

print(f"Neurons: {spks.shape[0]}, Trials: {len(responses)}")

# Extract trajectories
trajectories = []
for trial in range(spks.shape[1]):
    traj = spks[:, trial, 5:35].T
    trajectories.append(traj)

# PCA with more components
print("Fitting PCA with 6 components...")
all_data = np.vstack(trajectories)
pca = PCA(n_components=6)
pca.fit(all_data)
reduced_trajs = [pca.transform(t) for t in trajectories]

print(f"Explained variance per PC: {pca.explained_variance_ratio_}")
print(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

colors = {-1: 'blue', 0: 'gray', 1: 'red'}
labels = {-1: 'Left', 0: 'No-go', 1: 'Right'}

# =============================================================================
# Figure 1: Compare all PC pairs
# =============================================================================

fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
pc_pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (1, 3), (0, 3)]

for ax, (pc_x, pc_y) in zip(axes.flat, pc_pairs):
    for traj, resp in zip(reduced_trajs, responses):
        ax.plot(traj[:, pc_x], traj[:, pc_y], color=colors[resp], alpha=0.3, linewidth=0.5)
        # Mark start point
        ax.scatter(traj[0, pc_x], traj[0, pc_y], color=colors[resp], s=10, alpha=0.5)

    ax.set_xlabel(f'PC{pc_x+1}')
    ax.set_ylabel(f'PC{pc_y+1}')
    ax.set_title(f'PC{pc_x+1} vs PC{pc_y+1}')

# Add legend to first plot
for resp, color in colors.items():
    axes[0, 0].plot([], [], color=color, label=labels[resp], linewidth=2)
axes[0, 0].legend()

plt.suptitle('MOs Trajectories in Different PC Subspaces', fontsize=14)
plt.tight_layout()
fig1.savefig('trajectories_all_PCs.png', dpi=150, bbox_inches='tight')
print("Saved: trajectories_all_PCs.png")

# =============================================================================
# Figure 2: Direction field in PC3-PC4
# =============================================================================

# Use PC3-PC4 (indices 2-3)
trajs_pc34 = [t[:, 2:4] for t in reduced_trajs]

positions = []
velocities = []
resp_at_pos = []

for traj, resp in zip(trajs_pc34, responses):
    vel = np.diff(traj, axis=0)
    pos = traj[:-1]
    positions.append(pos)
    velocities.append(vel)
    resp_at_pos.extend([resp] * len(pos))

positions = np.vstack(positions)
velocities = np.vstack(velocities)
resp_at_pos = np.array(resp_at_pos)

# Create grid
grid_res = 20
bandwidth = 0.3
margin = 0.2

x_range = np.linspace(positions[:, 0].min() - margin,
                       positions[:, 0].max() + margin, grid_res)
y_range = np.linspace(positions[:, 1].min() - margin,
                       positions[:, 1].max() + margin, grid_res)
X, Y = np.meshgrid(x_range, y_range)

def compute_field(pos, vel, X, Y, bandwidth):
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            gp = np.array([X[i, j], Y[i, j]])
            dists = np.linalg.norm(pos - gp, axis=1)
            weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
            if weights.sum() > 1e-10:
                U[i, j] = np.average(vel[:, 0], weights=weights)
                V[i, j] = np.average(vel[:, 1], weights=weights)
    return U, V

print("Computing direction fields for PC3-PC4...")

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

for idx, resp_val in enumerate([-1, 0, 1]):
    ax = axes2[idx]
    mask = resp_at_pos == resp_val

    if mask.sum() > 50:
        U, V = compute_field(positions[mask], velocities[mask], X, Y, bandwidth)
        speed = np.sqrt(U**2 + V**2)
        ax.quiver(X, Y, U, V, speed, cmap='viridis', scale=20, width=0.004)

    for traj, r in zip(trajs_pc34, responses):
        if r == resp_val:
            ax.plot(traj[:, 0], traj[:, 1], color=colors[resp_val], alpha=0.3, linewidth=0.5)
            ax.scatter(traj[0, 0], traj[0, 1], color='green', s=15, alpha=0.5, zorder=5)

    ax.set_xlabel('PC3')
    ax.set_ylabel('PC4')
    ax.set_title(f'{labels[resp_val]} Trials')

plt.suptitle('Direction Field in PC3-PC4 Space (MOs)', fontsize=14)
plt.tight_layout()
fig2.savefig('direction_field_PC3_PC4.png', dpi=150, bbox_inches='tight')
print("Saved: direction_field_PC3_PC4.png")

# =============================================================================
# Figure 3: Mean trajectory comparison across PCs
# =============================================================================

trajs_by_resp = {-1: [], 1: []}
for traj, resp in zip(reduced_trajs, responses):
    if resp in trajs_by_resp:
        trajs_by_resp[resp].append(traj)

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
time_ms = np.arange(30) * 10 + 50

for pc_idx in range(6):
    ax = axes3.flat[pc_idx]

    for resp_val in [-1, 1]:
        trajs = trajs_by_resp[resp_val]
        if len(trajs) > 5:
            mean_pc = np.mean([t[:, pc_idx] for t in trajs], axis=0)
            std_pc = np.std([t[:, pc_idx] for t in trajs], axis=0)

            ax.plot(time_ms, mean_pc, color=colors[resp_val],
                   label=labels[resp_val], linewidth=2)
            ax.fill_between(time_ms, mean_pc - std_pc, mean_pc + std_pc,
                           color=colors[resp_val], alpha=0.2)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(f'PC{pc_idx+1}')
    ax.set_title(f'PC{pc_idx+1} over Time')
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    if pc_idx == 0:
        ax.legend()

plt.suptitle('Mean Trajectory per PC - Left vs Right (MOs)', fontsize=14)
plt.tight_layout()
fig3.savefig('mean_trajectory_all_PCs.png', dpi=150, bbox_inches='tight')
print("Saved: mean_trajectory_all_PCs.png")

# =============================================================================
# Quantify separation
# =============================================================================

print("\n" + "="*50)
print("Separation Analysis: Left vs Right")
print("="*50)

for pc_idx in range(6):
    left_trajs = [t[:, pc_idx] for t in trajs_by_resp[-1]]
    right_trajs = [t[:, pc_idx] for t in trajs_by_resp[1]]

    left_mean = np.mean(left_trajs, axis=0)
    right_mean = np.mean(right_trajs, axis=0)

    # Max separation
    separation = np.abs(left_mean - right_mean)
    max_sep = separation.max()
    max_sep_time = time_ms[separation.argmax()]

    # Pooled std for effect size
    left_std = np.std(left_trajs, axis=0)
    right_std = np.std(right_trajs, axis=0)
    pooled_std = np.sqrt((left_std**2 + right_std**2) / 2)
    effect_size = separation / (pooled_std + 1e-10)
    max_effect = effect_size.max()

    print(f"PC{pc_idx+1}: Max separation = {max_sep:.3f} at t={max_sep_time}ms, Effect size = {max_effect:.2f}")

print("\nDone!")
