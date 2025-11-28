"""
Direction Field with Quiver Plot - True Vector Field Visualization
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

session = alldat[0]  # Session with most VISp neurons
area_mask = session['brain_area'] == 'VISp'
spks = session['spks'][area_mask]
responses = session['response']

# Extract trajectories (time window 50-350ms)
print("Extracting trajectories...")
trajectories = []
for trial in range(spks.shape[1]):
    traj = spks[:, trial, 5:35].T  # (time, neurons)
    trajectories.append(traj)

# PCA to 2D
print("Fitting PCA...")
all_data = np.vstack(trajectories)
pca = PCA(n_components=2)
pca.fit(all_data)
reduced_trajs = [pca.transform(t) for t in trajectories]

print(f"Explained variance: {pca.explained_variance_ratio_}")

# Compute positions and velocities
print("Computing velocities...")
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

# Create grid for direction field
print("Creating direction field grid...")
grid_res = 20
margin = 0.1

x_range = np.linspace(positions[:, 0].min() - margin,
                       positions[:, 0].max() + margin, grid_res)
y_range = np.linspace(positions[:, 1].min() - margin,
                       positions[:, 1].max() + margin, grid_res)
X, Y = np.meshgrid(x_range, y_range)

# Estimate velocity at each grid point using kernel smoothing
U = np.zeros_like(X)
V = np.zeros_like(Y)
confidence = np.zeros_like(X)

bandwidth = 0.3

for i in range(grid_res):
    for j in range(grid_res):
        gp = np.array([X[i, j], Y[i, j]])
        dists = np.linalg.norm(positions - gp, axis=1)
        weights = np.exp(-0.5 * (dists / bandwidth) ** 2)

        if weights.sum() > 1e-10:
            U[i, j] = np.average(velocities[:, 0], weights=weights)
            V[i, j] = np.average(velocities[:, 1], weights=weights)
            confidence[i, j] = weights.sum()

# Normalize arrows for visibility
speed = np.sqrt(U**2 + V**2)
U_norm = U / (speed + 1e-10)
V_norm = V / (speed + 1e-10)

# Mask low confidence regions
mask = confidence < np.percentile(confidence[confidence > 0], 20)
U_norm[mask] = np.nan
V_norm[mask] = np.nan

# =============================================================================
# PLOTTING
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Direction field with quiver
ax1 = axes[0]
ax1.set_title('Direction Field (Quiver) - VISp', fontsize=14)

# Plot quiver (arrows)
q = ax1.quiver(X, Y, U_norm, V_norm, speed, cmap='viridis',
               scale=25, width=0.004, headwidth=4, headlength=5)
plt.colorbar(q, ax=ax1, label='Speed (magnitude)')

# Add some trajectories
colors = {-1: 'blue', 0: 'gray', 1: 'red'}
for traj, resp in list(zip(reduced_trajs, responses))[:30]:
    ax1.plot(traj[:, 0], traj[:, 1], color=colors[resp], alpha=0.4, linewidth=0.8)
    ax1.scatter(traj[0, 0], traj[0, 1], c='green', s=20, marker='o', zorder=5)
    ax1.scatter(traj[-1, 0], traj[-1, 1], c='black', s=20, marker='x', zorder=5)

ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')

# Legend
for resp, color in colors.items():
    label = {-1: 'Left', 0: 'No-go', 1: 'Right'}[resp]
    ax1.plot([], [], color=color, label=label, linewidth=2)
ax1.legend(loc='upper right')

# Plot 2: Streamlines (what we had before)
ax2 = axes[1]
ax2.set_title('Direction Field (Streamlines) - VISp', fontsize=14)

ax2.streamplot(X, Y, U, V, color=speed, cmap='viridis',
               density=1.5, linewidth=0.8, arrowsize=1)

for traj, resp in list(zip(reduced_trajs, responses))[:30]:
    ax2.plot(traj[:, 0], traj[:, 1], color=colors[resp], alpha=0.4, linewidth=0.8)

ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')

plt.tight_layout()
plt.savefig('direction_field_quiver.png', dpi=150, bbox_inches='tight')
print("\nSaved: direction_field_quiver.png")

# =============================================================================
# Additional: Separate by response type
# =============================================================================

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

for idx, (resp_val, title) in enumerate([(-1, 'Left Trials'), (0, 'No-go Trials'), (1, 'Right Trials')]):
    ax = axes2[idx]

    # Filter positions/velocities by response
    mask_resp = resp_at_pos == resp_val
    pos_resp = positions[mask_resp]
    vel_resp = velocities[mask_resp]

    if len(pos_resp) < 50:
        ax.set_title(f'{title} (insufficient data)')
        continue

    # Compute field for this response type
    U_resp = np.zeros_like(X)
    V_resp = np.zeros_like(Y)

    for i in range(grid_res):
        for j in range(grid_res):
            gp = np.array([X[i, j], Y[i, j]])
            dists = np.linalg.norm(pos_resp - gp, axis=1)
            weights = np.exp(-0.5 * (dists / bandwidth) ** 2)

            if weights.sum() > 1e-10:
                U_resp[i, j] = np.average(vel_resp[:, 0], weights=weights)
                V_resp[i, j] = np.average(vel_resp[:, 1], weights=weights)

    speed_resp = np.sqrt(U_resp**2 + V_resp**2)

    ax.quiver(X, Y, U_resp, V_resp, speed_resp, cmap='viridis',
              scale=25, width=0.004)

    # Plot trajectories for this response
    color = colors[resp_val]
    for traj, r in zip(reduced_trajs, responses):
        if r == resp_val:
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.3, linewidth=0.5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)

plt.tight_layout()
plt.savefig('direction_field_by_response.png', dpi=150, bbox_inches='tight')
print("Saved: direction_field_by_response.png")

print("\nDone!")
