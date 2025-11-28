"""
Direction Fields for Neural Dynamics
====================================
Project D: Extracting direction fields from neural population activity
without assuming a specific functional form.

The idea: Instead of finding the equation dx/dt = f(x),
we reconstruct the direction field directly from data:
For each point x in state space -> what is the average direction of movement?

This can reveal:
- Attractors (stable states)
- Saddle points
- Limit cycles
- And relate them to behavior
"""

import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PART 1: Load and Prepare Data
# =============================================================================

def load_steinmetz_data(data_dir='DATA'):
    """Load all Steinmetz sessions"""
    alldat = []
    for i in range(1, 4):
        fname = os.path.join(data_dir, f'steinmetz_part{i}.npy')
        dat = np.load(fname, allow_pickle=True)['dat']
        alldat.extend(dat)
    return alldat


def select_session(alldat, min_neurons=50, area='VISp'):
    """Select a session with enough neurons in target area"""
    best_idx = None
    max_neurons = 0

    for idx, session in enumerate(alldat):
        n_area = np.sum(session['brain_area'] == area)
        n_trials = len(session['response'])
        if n_area > max_neurons and n_trials > 150:
            max_neurons = n_area
            best_idx = idx

    return best_idx, max_neurons


def prepare_trajectories(session, area='VISp', time_range=(0, 50)):
    """
    Extract neural trajectories from a session.

    Returns:
        trajectories: list of arrays, each (n_timepoints, n_neurons)
        responses: behavioral response for each trial
    """
    area_mask = session['brain_area'] == area
    spks = session['spks'][area_mask]  # (neurons, trials, time)

    n_neurons, n_trials, n_time = spks.shape
    t_start, t_end = time_range

    trajectories = []
    responses = []

    for trial in range(n_trials):
        # Get spike counts over time for this trial
        traj = spks[:, trial, t_start:t_end].T  # (time, neurons)
        trajectories.append(traj)
        responses.append(session['response'][trial])

    return trajectories, np.array(responses)


# =============================================================================
# PART 2: Dimensionality Reduction
# =============================================================================

def fit_pca(trajectories, n_components=3):
    """
    Fit PCA on concatenated trajectories.

    Returns:
        pca: fitted PCA object
        reduced_trajectories: list of trajectories in reduced space
    """
    # Concatenate all trajectories for fitting
    all_data = np.vstack(trajectories)

    pca = PCA(n_components=n_components)
    pca.fit(all_data)

    # Transform each trajectory
    reduced_trajectories = [pca.transform(traj) for traj in trajectories]

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total explained: {sum(pca.explained_variance_ratio_):.1%}")

    return pca, reduced_trajectories


# =============================================================================
# PART 3: Compute Direction Field
# =============================================================================

def compute_velocities(trajectories):
    """
    Compute velocity vectors (dr/dt) for each point in trajectories.

    Returns:
        positions: array of all positions (N, dim)
        velocities: array of all velocities (N, dim)
    """
    positions = []
    velocities = []

    for traj in trajectories:
        # Velocity is the difference between consecutive timepoints
        vel = np.diff(traj, axis=0)
        pos = traj[:-1]  # Position at each velocity measurement

        positions.append(pos)
        velocities.append(vel)

    positions = np.vstack(positions)
    velocities = np.vstack(velocities)

    return positions, velocities


def estimate_direction_field(positions, velocities, grid_resolution=20,
                             smoothing='kde', bandwidth=0.5):
    """
    Estimate the direction field on a grid.

    For each grid point, estimate the average velocity of nearby trajectories.

    Args:
        positions: (N, dim) array of trajectory positions
        velocities: (N, dim) array of velocities at each position
        grid_resolution: number of grid points per dimension
        smoothing: 'kde' for kernel density estimation, 'nearest' for nearest neighbors
        bandwidth: smoothing parameter

    Returns:
        grid_points: meshgrid of positions
        field_vectors: velocity vectors at each grid point
    """
    dim = positions.shape[1]

    # Create grid
    grid_ranges = []
    for d in range(dim):
        margin = 0.1 * (positions[:, d].max() - positions[:, d].min())
        grid_ranges.append(np.linspace(
            positions[:, d].min() - margin,
            positions[:, d].max() + margin,
            grid_resolution
        ))

    if dim == 2:
        X, Y = np.meshgrid(grid_ranges[0], grid_ranges[1])
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
    elif dim == 3:
        X, Y, Z = np.meshgrid(grid_ranges[0], grid_ranges[1], grid_ranges[2])
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    else:
        raise ValueError("Only 2D and 3D supported for visualization")

    # Estimate velocity at each grid point using kernel smoothing
    field_vectors = np.zeros_like(grid_points)
    weights_sum = np.zeros(len(grid_points))

    for i, gp in enumerate(grid_points):
        # Compute distances to all data points
        dists = np.linalg.norm(positions - gp, axis=1)

        # Kernel weights (Gaussian)
        weights = np.exp(-0.5 * (dists / bandwidth) ** 2)

        # Weighted average of velocities
        if weights.sum() > 1e-10:
            field_vectors[i] = np.average(velocities, axis=0, weights=weights)
            weights_sum[i] = weights.sum()
        else:
            field_vectors[i] = np.nan
            weights_sum[i] = 0

    return grid_points, field_vectors, weights_sum


def find_fixed_points(grid_points, field_vectors, weights_sum, threshold=0.1):
    """
    Find candidate fixed points (where velocity is near zero).

    Returns:
        fixed_points: array of candidate fixed point locations
        stability: estimate of stability (negative = stable attractor)
    """
    # Only consider points with enough data
    valid = weights_sum > np.percentile(weights_sum[weights_sum > 0], 25)

    # Speed at each grid point
    speeds = np.linalg.norm(field_vectors, axis=1)
    speeds[~valid] = np.inf

    # Find local minima of speed
    min_speed = np.nanpercentile(speeds[valid], 10)
    candidates = speeds < min_speed * 2

    fixed_points = grid_points[candidates & valid]

    return fixed_points


# =============================================================================
# PART 4: Visualization
# =============================================================================

def plot_direction_field_2d(grid_points, field_vectors, weights_sum,
                            trajectories=None, responses=None,
                            fixed_points=None, ax=None, title='Direction Field'):
    """
    Plot 2D direction field with optional trajectories colored by response.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Reshape for quiver plot
    n = int(np.sqrt(len(grid_points)))
    X = grid_points[:, 0].reshape(n, n)
    Y = grid_points[:, 1].reshape(n, n)
    U = field_vectors[:, 0].reshape(n, n)
    V = field_vectors[:, 1].reshape(n, n)

    # Mask low-confidence regions
    W = weights_sum.reshape(n, n)
    mask = W < np.percentile(W[W > 0], 20)
    U[mask] = np.nan
    V[mask] = np.nan

    # Plot direction field
    speed = np.sqrt(U**2 + V**2)
    ax.streamplot(X, Y, U, V, color=speed, cmap='viridis',
                  density=1.5, linewidth=0.8, arrowsize=1)

    # Plot trajectories colored by response
    if trajectories is not None and responses is not None:
        colors = {-1: 'blue', 0: 'gray', 1: 'red'}
        labels = {-1: 'Left', 0: 'No-go', 1: 'Right'}

        for traj, resp in zip(trajectories, responses):
            ax.plot(traj[:, 0], traj[:, 1], color=colors[resp],
                   alpha=0.3, linewidth=0.5)

        # Legend
        for resp, color in colors.items():
            ax.plot([], [], color=color, label=labels[resp], linewidth=2)
        ax.legend(loc='upper right')

    # Plot fixed points
    if fixed_points is not None and len(fixed_points) > 0:
        ax.scatter(fixed_points[:, 0], fixed_points[:, 1],
                  c='yellow', s=200, marker='*', edgecolors='black',
                  linewidths=2, zorder=10, label='Fixed points')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)

    return ax


def plot_3d_trajectories(trajectories, responses, title='Neural Trajectories in PC Space'):
    """Plot 3D trajectories colored by behavioral response"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = {-1: 'blue', 0: 'gray', 1: 'red'}

    for traj, resp in zip(trajectories, responses):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
               color=colors[resp], alpha=0.3, linewidth=0.5)

    # Mark start and end points
    for traj, resp in zip(trajectories[:10], responses[:10]):
        ax.scatter(*traj[0], c='green', s=50, marker='o')  # Start
        ax.scatter(*traj[-1], c='black', s=50, marker='x')  # End

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)

    return fig, ax


# =============================================================================
# PART 5: Analysis - Relate Attractors to Behavior
# =============================================================================

def analyze_trajectory_endpoints(trajectories, responses, n_clusters=3):
    """
    Analyze where trajectories end up based on behavioral response.

    If different responses lead to different attractor basins,
    this suggests the attractors are behaviorally meaningful.
    """
    from sklearn.cluster import KMeans

    # Get endpoint of each trajectory
    endpoints = np.array([traj[-1] for traj in trajectories])

    # Cluster endpoints
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(endpoints)

    # Analyze response distribution per cluster
    print("\nEndpoint Analysis:")
    print("=" * 50)

    for c in range(n_clusters):
        mask = clusters == c
        resp_in_cluster = responses[mask]

        n_left = np.sum(resp_in_cluster == -1)
        n_nogo = np.sum(resp_in_cluster == 0)
        n_right = np.sum(resp_in_cluster == 1)
        total = len(resp_in_cluster)

        print(f"\nCluster {c} ({total} trials):")
        print(f"  Left:   {n_left:3d} ({100*n_left/total:.1f}%)")
        print(f"  No-go:  {n_nogo:3d} ({100*n_nogo/total:.1f}%)")
        print(f"  Right:  {n_right:3d} ({100*n_right/total:.1f}%)")

    return clusters, kmeans.cluster_centers_


def compute_trajectory_divergence(trajectories, responses, time_point=10):
    """
    Compute when trajectories for different responses start to diverge.
    """
    # Separate by response
    left_trajs = [t for t, r in zip(trajectories, responses) if r == -1]
    right_trajs = [t for t, r in zip(trajectories, responses) if r == 1]

    if len(left_trajs) < 5 or len(right_trajs) < 5:
        return None

    # Compute mean trajectory for each response
    min_len = min(min(len(t) for t in left_trajs), min(len(t) for t in right_trajs))

    left_mean = np.mean([t[:min_len] for t in left_trajs], axis=0)
    right_mean = np.mean([t[:min_len] for t in right_trajs], axis=0)

    # Distance between mean trajectories over time
    divergence = np.linalg.norm(left_mean - right_mean, axis=1)

    return divergence, left_mean, right_mean


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Direction Fields for Neural Dynamics")
    print("=" * 60)

    # Load data
    print("\n1. Loading Steinmetz data...")
    alldat = load_steinmetz_data()
    print(f"   Loaded {len(alldat)} sessions")

    # Select best session
    print("\n2. Selecting session with most VISp neurons...")
    session_idx, n_neurons = select_session(alldat, area='VISp')
    session = alldat[session_idx]
    print(f"   Session {session_idx}: {n_neurons} VISp neurons, "
          f"{len(session['response'])} trials")
    print(f"   Mouse: {session['mouse_name']}, Date: {session['date_exp']}")

    # Extract trajectories
    print("\n3. Extracting neural trajectories...")
    trajectories, responses = prepare_trajectories(
        session, area='VISp', time_range=(5, 35)  # 50-350ms post-stimulus
    )
    print(f"   {len(trajectories)} trajectories extracted")
    print(f"   Response distribution: Left={sum(responses==-1)}, "
          f"No-go={sum(responses==0)}, Right={sum(responses==1)}")

    # PCA
    print("\n4. Fitting PCA...")
    pca, reduced_trajectories = fit_pca(trajectories, n_components=3)

    # Compute velocities
    print("\n5. Computing velocity vectors...")
    # Use 2D for visualization
    trajs_2d = [t[:, :2] for t in reduced_trajectories]
    positions, velocities = compute_velocities(trajs_2d)
    print(f"   {len(positions)} position-velocity pairs")

    # Estimate direction field
    print("\n6. Estimating direction field...")
    grid_points, field_vectors, weights_sum = estimate_direction_field(
        positions, velocities, grid_resolution=25, bandwidth=0.5
    )

    # Find fixed points
    print("\n7. Finding candidate fixed points...")
    fixed_points = find_fixed_points(grid_points, field_vectors, weights_sum)
    print(f"   Found {len(fixed_points)} candidate fixed points")

    # Analyze endpoints
    print("\n8. Analyzing trajectory endpoints...")
    clusters, centers = analyze_trajectory_endpoints(
        reduced_trajectories, responses, n_clusters=3
    )

    # Compute divergence
    print("\n9. Computing trajectory divergence...")
    result = compute_trajectory_divergence(reduced_trajectories, responses)
    if result:
        divergence, left_mean, right_mean = result
        print(f"   Divergence at t=0: {divergence[0]:.3f}")
        print(f"   Divergence at t=end: {divergence[-1]:.3f}")
        print(f"   Max divergence: {divergence.max():.3f} at t={divergence.argmax()}")

    # Create visualizations
    print("\n10. Creating visualizations...")

    # Figure 1: 2D Direction Field
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    plot_direction_field_2d(
        grid_points, field_vectors, weights_sum,
        trajectories=trajs_2d, responses=responses,
        fixed_points=fixed_points, ax=ax1,
        title=f'Direction Field - VISp ({session["mouse_name"]})'
    )
    fig1.savefig('direction_field_2d.png', dpi=150, bbox_inches='tight')
    print("   Saved: direction_field_2d.png")

    # Figure 2: 3D Trajectories
    fig2, ax2 = plot_3d_trajectories(
        reduced_trajectories, responses,
        title=f'Neural Trajectories in PC Space - {session["mouse_name"]}'
    )
    fig2.savefig('trajectories_3d.png', dpi=150, bbox_inches='tight')
    print("   Saved: trajectories_3d.png")

    # Figure 3: Divergence over time
    if result:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        time_ms = np.arange(len(divergence)) * 10 + 50  # Convert to ms
        ax3.plot(time_ms, divergence, 'k-', linewidth=2)
        ax3.axhline(y=0, color='gray', linestyle='--')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Distance between Left and Right trajectories')
        ax3.set_title('Trajectory Divergence by Response Type')
        fig3.savefig('trajectory_divergence.png', dpi=150, bbox_inches='tight')
        print("   Saved: trajectory_divergence.png")

    # Figure 4: Mean trajectories
    if result:
        fig4, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PC1 over time
        axes[0].plot(time_ms, left_mean[:, 0], 'b-', label='Left', linewidth=2)
        axes[0].plot(time_ms, right_mean[:, 0], 'r-', label='Right', linewidth=2)
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('PC1')
        axes[0].set_title('Mean Trajectory - PC1')
        axes[0].legend()

        # PC2 over time
        axes[1].plot(time_ms, left_mean[:, 1], 'b-', label='Left', linewidth=2)
        axes[1].plot(time_ms, right_mean[:, 1], 'r-', label='Right', linewidth=2)
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('PC2')
        axes[1].set_title('Mean Trajectory - PC2')
        axes[1].legend()

        fig4.savefig('mean_trajectories.png', dpi=150, bbox_inches='tight')
        print("   Saved: mean_trajectories.png")

    # plt.show()  # Commented out for non-interactive mode

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - direction_field_2d.png")
    print("  - trajectories_3d.png")
    print("  - trajectory_divergence.png")
    print("  - mean_trajectories.png")
