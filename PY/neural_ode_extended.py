"""
Neural ODE vs GLM - Extended Analysis
=====================================
Test across multiple sessions and brain areas to verify the result.
"""

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("Loading data...")
data_dir = 'DATA'
alldat = []
for i in range(1, 4):
    fname = os.path.join(data_dir, f'steinmetz_part{i}.npy')
    dat = np.load(fname, allow_pickle=True)['dat']
    alldat.extend(dat)

# Models
class NeuralODE(nn.Module):
    def __init__(self, n_dims, hidden_size=64):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(n_dims, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_dims)
        )

    def forward(self, x):
        dx = self.func(x)
        x_next = x + dx
        return torch.relu(x_next)


def train_and_evaluate(spks, n_epochs=30):
    """Train GLM and Neural ODE, return metrics"""
    n_neurons, n_trials, n_time = spks.shape

    # Prepare data
    X_list, Y_list = [], []
    for trial in range(n_trials):
        for t in range(10, n_time - 1):
            X_list.append(spks[:, trial, t])
            Y_list.append(spks[:, trial, t + 1])

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    # Split
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    X_train, X_test = X[indices[:n_train]], X[indices[n_train:]]
    Y_train, Y_test = Y[indices[:n_train]], Y[indices[n_train:]]

    # === GLM ===
    glm_pred = np.zeros_like(Y_test)
    for neuron_idx in range(n_neurons):
        y_train = Y_train[:, neuron_idx]
        if y_train.sum() < 10:
            glm_pred[:, neuron_idx] = 0
            continue
        try:
            glm = PoissonRegressor(alpha=0.1, max_iter=100)
            glm.fit(X_train, y_train)
            glm_pred[:, neuron_idx] = glm.predict(X_test)
        except:
            glm_pred[:, neuron_idx] = y_train.mean()

    glm_mse = mean_squared_error(Y_test, glm_pred)
    glm_r2 = r2_score(Y_test.flatten(), glm_pred.flatten())

    # === Neural ODE ===
    X_train_t = torch.FloatTensor(X_train).to(device)
    Y_train_t = torch.FloatTensor(Y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t),
                              batch_size=256, shuffle=True)

    model = NeuralODE(n_dims=n_neurons, hidden_size=min(128, n_neurons*2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        node_pred = model(X_test_t).cpu().numpy()

    node_mse = mean_squared_error(Y_test, node_pred)
    node_r2 = r2_score(Y_test.flatten(), node_pred.flatten())

    return {
        'glm_mse': glm_mse, 'glm_r2': glm_r2,
        'node_mse': node_mse, 'node_r2': node_r2,
        'n_neurons': n_neurons, 'n_samples': n_samples
    }


# =============================================================================
# Run on multiple sessions and areas
# =============================================================================

areas_to_test = ['VISp', 'MOs', 'MOp', 'ACA', 'SC']
results = []

print("\n" + "="*70)
print("Testing Neural ODE vs GLM across sessions and areas")
print("="*70)

for area in areas_to_test:
    print(f"\n--- Area: {area} ---")

    # Find sessions with enough neurons in this area
    for session_idx, session in enumerate(alldat):
        area_mask = session['brain_area'] == area
        n_neurons = np.sum(area_mask)

        if n_neurons >= 20 and len(session['response']) >= 150:
            spks = session['spks'][area_mask]

            print(f"  Session {session_idx} ({session['mouse_name']}): {n_neurons} neurons...", end=" ")

            try:
                metrics = train_and_evaluate(spks, n_epochs=30)
                metrics['session'] = session_idx
                metrics['area'] = area
                metrics['mouse'] = session['mouse_name']

                improvement = (metrics['glm_mse'] - metrics['node_mse']) / metrics['glm_mse'] * 100

                if metrics['node_mse'] < metrics['glm_mse']:
                    print(f"NODE wins! ({improvement:.1f}% better)")
                else:
                    print(f"GLM wins ({-improvement:.1f}% better)")

                results.append(metrics)

                # Limit to 2 sessions per area for speed
                if len([r for r in results if r['area'] == area]) >= 2:
                    break

            except Exception as e:
                print(f"Error: {e}")
                continue

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n{'Area':<8} {'Session':<10} {'Mouse':<12} {'Neurons':<8} {'GLM MSE':<10} {'NODE MSE':<10} {'Winner':<8} {'Improv':<8}")
print("-"*74)

node_wins = 0
glm_wins = 0

for r in results:
    winner = "NODE" if r['node_mse'] < r['glm_mse'] else "GLM"
    if winner == "NODE":
        node_wins += 1
    else:
        glm_wins += 1

    improvement = (r['glm_mse'] - r['node_mse']) / r['glm_mse'] * 100

    print(f"{r['area']:<8} {r['session']:<10} {r['mouse']:<12} {r['n_neurons']:<8} "
          f"{r['glm_mse']:<10.4f} {r['node_mse']:<10.4f} {winner:<8} {improvement:>+.1f}%")

print("-"*74)
print(f"\nNeural ODE wins: {node_wins}/{len(results)} ({100*node_wins/len(results):.0f}%)")
print(f"GLM wins: {glm_wins}/{len(results)} ({100*glm_wins/len(results):.0f}%)")

# Average improvement
avg_glm_mse = np.mean([r['glm_mse'] for r in results])
avg_node_mse = np.mean([r['node_mse'] for r in results])
avg_improvement = (avg_glm_mse - avg_node_mse) / avg_glm_mse * 100

print(f"\nAverage MSE - GLM: {avg_glm_mse:.4f}, Neural ODE: {avg_node_mse:.4f}")
print(f"Average improvement: {avg_improvement:.1f}%")

# R2 comparison
avg_glm_r2 = np.mean([r['glm_r2'] for r in results])
avg_node_r2 = np.mean([r['node_r2'] for r in results])
print(f"\nAverage R2 - GLM: {avg_glm_r2:.4f}, Neural ODE: {avg_node_r2:.4f}")

# =============================================================================
# Visualization
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# MSE comparison
areas = [r['area'] for r in results]
glm_mses = [r['glm_mse'] for r in results]
node_mses = [r['node_mse'] for r in results]

x = np.arange(len(results))
width = 0.35

axes[0].bar(x - width/2, glm_mses, width, label='GLM', color='blue', alpha=0.7)
axes[0].bar(x + width/2, node_mses, width, label='Neural ODE', color='green', alpha=0.7)
axes[0].set_ylabel('MSE (lower is better)')
axes[0].set_title('MSE Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"{r['area']}\n{r['session']}" for r in results], rotation=45, ha='right')
axes[0].legend()

# R2 comparison
glm_r2s = [r['glm_r2'] for r in results]
node_r2s = [r['node_r2'] for r in results]

axes[1].bar(x - width/2, glm_r2s, width, label='GLM', color='blue', alpha=0.7)
axes[1].bar(x + width/2, node_r2s, width, label='Neural ODE', color='green', alpha=0.7)
axes[1].set_ylabel('R² (higher is better)')
axes[1].set_title('R² Comparison')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{r['area']}\n{r['session']}" for r in results], rotation=45, ha='right')
axes[1].legend()

# Improvement by area
area_improvements = {}
for r in results:
    area = r['area']
    imp = (r['glm_mse'] - r['node_mse']) / r['glm_mse'] * 100
    if area not in area_improvements:
        area_improvements[area] = []
    area_improvements[area].append(imp)

area_names = list(area_improvements.keys())
area_means = [np.mean(area_improvements[a]) for a in area_names]
colors = ['green' if m > 0 else 'red' for m in area_means]

axes[2].bar(area_names, area_means, color=colors, alpha=0.7)
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[2].set_ylabel('Improvement over GLM (%)')
axes[2].set_title('Neural ODE Improvement by Brain Area')

plt.tight_layout()
plt.savefig('neural_ode_extended_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: neural_ode_extended_results.png")

# Statistical test
from scipy import stats
improvements = [(r['glm_mse'] - r['node_mse']) / r['glm_mse'] * 100 for r in results]
t_stat, p_value = stats.ttest_1samp(improvements, 0)
print(f"\nStatistical test (improvement > 0): t={t_stat:.2f}, p={p_value:.4f}")

if p_value < 0.05:
    print("Result: Neural ODE significantly outperforms GLM!")
else:
    print("Result: Difference not statistically significant")

print("\nDone!")
