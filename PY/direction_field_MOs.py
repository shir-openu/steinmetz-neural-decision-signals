"""
Direction Field Analysis - MOs (Secondary Motor Cortex)
Compare to VISp - does motor cortex show response-specific dynamics?
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

# Find session with most MOs neurons
best_idx = None
max_neurons = 0
for idx, session in enumerate(alldat):
    n_mos = np.sum(session['brain_area'] == 'MOs')
    n_trials = len(session['response'])
    if n_mos > max_neurons and n_trials > 150:
        max_neurons = n_mos
        best_idx = idx

print(f"Best session for MOs: {best_idx} with {max_neurons} neurons")

session = alldat[best_idx]
print(f"Mouse: {session['mouse_name']}, Date: {session['date_exp']}")

area_mask = session['brain_area'] == 'MOs'
spks = session['spks'][area_mask]
responses = session['response']

print(f"Trials: {len(responses)} (Left={sum(responses==-1)}, No-go={sum(responses==0)}, Right={sum(responses==1)})")

# Extract trajectories
print("Extracting trajectories...")
trajectories = []
for trial in range(spks.shape[1]):
    traj = spks[:, trial, 5:35].T  # 50-350ms
    trajectories.append(traj)

# PCA
print("Fitting PCA...")
all_data = np.vstack(trajectories)
pca = PCA(n_components=2)
pca.fit(all_data)
reduced_trajs = [pca.transform(t) for t in trajectories]

print(f"Explained variance: {pca.explained_variance_ratio_} = {sum(pca.explained_variance_ratio_):.1%}")

# Compute velocities
positions = []
velocities = []
resp_at_pos = []

for traj, resp in zip(reduced_trajs, responses):
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
margin = 0.2
bandwidth = 0.4

x_range = np.linspace(positions[:, 0].min() - margin,
                       positions[:, 0].max() + margin, grid_res)
y_range = np.linspace(positions[:, 1].min() - margin,
                       positions[:, 1].max() + margin, grid_res)
X, Y = np.meshgrid(x_range, y_range)

def compute_field(pos, vel, X, Y, bandwidth):
    """Compute direction field using kernel smoothing"""
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    confidence = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            gp = np.array([X[i, j], Y[i, j]])
            dists = np.linalg.norm(pos - gp, axis=1)
            weights = np.exp(-0.5 * (dists / bandwidth) ** 2)

            if weights.sum() > 1e-10:
                U[i, j] = np.average(vel[:, 0], weights=weights)
                V[i, j] = np.average(vel[:, 1], weights=weights)
                confidence[i, j] = weights.sum()

    return U, V, confidence

# Compute overall field
print("Computing direction fields...")
U_all, V_all, conf_all = compute_field(positions, velocities, X, Y, bandwidth)
speed_all = np.sqrt(U_all**2 + V_all**2)

# Compute field per response type
fields_by_resp = {}
for resp_val in [-1, 0, 1]:
    mask = resp_at_pos == resp_val
    if mask.sum() > 50:
        U, V, conf = compute_field(positions[mask], velocities[mask], X, Y, bandwidth)
        fields_by_resp[resp_val] = (U, V, conf)

# =============================================================================
# PLOTTING
# =============================================================================

colors = {-1: 'blue', 0: 'gray', 1: 'red'}
labels = {-1: 'Left', 0: 'No-go', 1: 'Right'}

# Figure 1: Overall direction field
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.set_title(f'Direction Field - MOs ({session["mouse_name"]})', fontsize=14)

q = ax1.quiver(X, Y, U_all, V_all, speed_all, cmap='viridis',
               scale=30, width=0.004, headwidth=4)
plt.colorbar(q, ax=ax1, label='Speed')

for traj, resp in list(zip(reduced_trajs, responses))[:50]:
    ax1.plot(traj[:, 0], traj[:, 1], color=colors[resp], alpha=0.3, linewidth=0.7)

ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
for resp, color in colors.items():
    ax1.plot([], [], color=color, label=labels[resp], linewidth=2)
ax1.legend()

fig1.savefig('direction_field_MOs.png', dpi=150, bbox_inches='tight')
print("Saved: direction_field_MOs.png")

# Figure 2: Separate by response
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

for idx, resp_val in enumerate([-1, 0, 1]):
    ax = axes2[idx]
    ax.set_title(f'{labels[resp_val]} Trials - MOs')

    if resp_val in fields_by_resp:
        U, V, conf = fields_by_resp[resp_val]
        speed = np.sqrt(U**2 + V**2)
        ax.quiver(X, Y, U, V, speed, cmap='viridis', scale=30, width=0.004)

    for traj, r in zip(reduced_trajs, responses):
        if r == resp_val:
            ax.plot(traj[:, 0], traj[:, 1], color=colors[resp_val], alpha=0.3, linewidth=0.5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

fig2.savefig('direction_field_MOs_by_response.png', dpi=150, bbox_inches='tight')
print("Saved: direction_field_MOs_by_response.png")

# Figure 3: Compare mean trajectories
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

# Separate trajectories by response
trajs_by_resp = {-1: [], 0: [], 1: []}
for traj, resp in zip(reduced_trajs, responses):
    trajs_by_resp[resp].append(traj)

# Compute mean trajectories
time_ms = np.arange(29) * 10 + 50  # 50-340ms

for resp_val in [-1, 1]:  # Left and Right only
    if len(trajs_by_resp[resp_val]) > 5:
        mean_traj = np.mean(trajs_by_resp[resp_val], axis=0)
        std_traj = np.std(trajs_by_resp[resp_val], axis=0)

        # PC1 over time
        axes3[0].plot(time_ms, mean_traj[:, 0], color=colors[resp_val],
                     label=labels[resp_val], linewidth=2)
        axes3[0].fill_between(time_ms, mean_traj[:, 0] - std_traj[:, 0],
                              mean_traj[:, 0] + std_traj[:, 0],
                              color=colors[resp_val], alpha=0.2)

        # PC2 over time
        axes3[1].plot(time_ms, mean_traj[:, 1], color=colors[resp_val],
                     label=labels[resp_val], linewidth=2)
        axes3[1].fill_between(time_ms, mean_traj[:, 1] - std_traj[:, 1],
                              mean_traj[:, 1] + std_traj[:, 1],
                              color=colors[resp_val], alpha=0.2)

axes3[0].set_xlabel('Time (ms)')
axes3[0].set_ylabel('PC1')
axes3[0].set_title('Mean Trajectory - PC1 (MOs)')
axes3[0].legend()
axes3[0].axvline(x=100, color='gray', linestyle='--', alpha=0.5)

axes3[1].set_xlabel('Time (ms)')
axes3[1].set_ylabel('PC2')
axes3[1].set_title('Mean Trajectory - PC2 (MOs)')
axes3[1].legend()
axes3[1].axvline(x=100, color='gray', linestyle='--', alpha=0.5)

fig3.savefig('mean_trajectories_MOs.png', dpi=150, bbox_inches='tight')
print("Saved: mean_trajectories_MOs.png")

# Compute divergence
if len(trajs_by_resp[-1]) > 5 and len(trajs_by_resp[1]) > 5:
    left_mean = np.mean(trajs_by_resp[-1], axis=0)
    right_mean = np.mean(trajs_by_resp[1], axis=0)
    divergence = np.linalg.norm(left_mean - right_mean, axis=1)

    print(f"\nTrajectory Divergence (MOs):")
    print(f"  At t=50ms:  {divergence[0]:.3f}")
    print(f"  At t=200ms: {divergence[15]:.3f}")
    print(f"  At t=340ms: {divergence[-1]:.3f}")
    print(f"  Maximum: {divergence.max():.3f} at t={time_ms[divergence.argmax()]}ms")

print("\nDone!")
