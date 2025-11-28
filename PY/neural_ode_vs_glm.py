"""
Neural ODE vs GLM for Spike Train Prediction
=============================================
Proof of concept: Can numerical methods (Neural ODE) predict neural dynamics
better than analytical methods (GLM)?

Task: Given neural activity at time t, predict activity at time t+1
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

# Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# PART 1: Load and Prepare Data
# =============================================================================

print("\n" + "="*60)
print("Loading Steinmetz data...")
print("="*60)

data_dir = 'DATA'
alldat = []
for i in range(1, 4):
    fname = os.path.join(data_dir, f'steinmetz_part{i}.npy')
    dat = np.load(fname, allow_pickle=True)['dat']
    alldat.extend(dat)

# Use session with many neurons
session = alldat[11]  # Good session
area = 'VISp'
area_mask = session['brain_area'] == area
spks = session['spks'][area_mask]
n_neurons, n_trials, n_time = spks.shape

print(f"Session: {session['mouse_name']}, {session['date_exp']}")
print(f"Area: {area}, Neurons: {n_neurons}, Trials: {n_trials}, Time bins: {n_time}")

# Prepare data: predict next time step from current
# X = activity at time t, Y = activity at time t+1
X_list = []
Y_list = []

for trial in range(n_trials):
    for t in range(10, n_time - 1):  # Start from t=10 to have some history
        X_list.append(spks[:, trial, t])
        Y_list.append(spks[:, trial, t + 1])

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)

print(f"Data shape: X={X.shape}, Y={Y.shape}")

# Train/test split
n_samples = len(X)
n_train = int(0.8 * n_samples)
indices = np.random.permutation(n_samples)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# =============================================================================
# PART 2: GLM Baseline (Poisson Regression)
# =============================================================================

print("\n" + "="*60)
print("Training GLM (Poisson Regression)...")
print("="*60)

# Train one GLM per neuron
glm_predictions = np.zeros_like(Y_test)

for neuron_idx in range(n_neurons):
    if neuron_idx % 20 == 0:
        print(f"  Neuron {neuron_idx}/{n_neurons}")

    # Target for this neuron
    y_train = Y_train[:, neuron_idx]
    y_test = Y_test[:, neuron_idx]

    # Skip if all zeros
    if y_train.sum() < 10:
        glm_predictions[:, neuron_idx] = 0
        continue

    try:
        glm = PoissonRegressor(alpha=0.1, max_iter=200)
        glm.fit(X_train, y_train)
        glm_predictions[:, neuron_idx] = glm.predict(X_test)
    except:
        glm_predictions[:, neuron_idx] = y_train.mean()

# Evaluate GLM
glm_mse = mean_squared_error(Y_test, glm_predictions)
glm_r2 = r2_score(Y_test.flatten(), glm_predictions.flatten())
print(f"\nGLM Results:")
print(f"  MSE: {glm_mse:.4f}")
print(f"  R2:  {glm_r2:.4f}")

# =============================================================================
# PART 3: Neural ODE
# =============================================================================

print("\n" + "="*60)
print("Training Neural ODE...")
print("="*60)

# Simple Neural ODE implementation
class NeuralODEFunc(nn.Module):
    """The function f(x) in dx/dt = f(x)"""
    def __init__(self, n_dims, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_dims, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_dims)
        )

    def forward(self, x):
        return self.net(x)


class NeuralODE(nn.Module):
    """Neural ODE: x(t+1) = x(t) + f(x(t)) * dt"""
    def __init__(self, n_dims, hidden_size=64):
        super().__init__()
        self.func = NeuralODEFunc(n_dims, hidden_size)
        self.dt = 1.0  # Time step

    def forward(self, x):
        # Euler integration: x_next = x + f(x) * dt
        dx = self.func(x)
        x_next = x + dx * self.dt
        # Ensure non-negative (spike counts)
        x_next = torch.relu(x_next)
        return x_next


# Prepare PyTorch data
X_train_t = torch.FloatTensor(X_train).to(device)
Y_train_t = torch.FloatTensor(Y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
Y_test_t = torch.FloatTensor(Y_test).to(device)

train_dataset = TensorDataset(X_train_t, Y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Initialize model
model = NeuralODE(n_dims=n_neurons, hidden_size=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
n_epochs = 50
losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

# Evaluate Neural ODE
model.eval()
with torch.no_grad():
    node_predictions = model(X_test_t).cpu().numpy()

node_mse = mean_squared_error(Y_test, node_predictions)
node_r2 = r2_score(Y_test.flatten(), node_predictions.flatten())
print(f"\nNeural ODE Results:")
print(f"  MSE: {node_mse:.4f}")
print(f"  R2:  {node_r2:.4f}")

# =============================================================================
# PART 4: Simple MLP Baseline (for comparison)
# =============================================================================

print("\n" + "="*60)
print("Training Simple MLP (non-ODE baseline)...")
print("="*60)

class SimpleMLP(nn.Module):
    def __init__(self, n_dims, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_dims),
            nn.ReLU()  # Non-negative outputs
        )

    def forward(self, x):
        return self.net(x)

mlp_model = SimpleMLP(n_dims=n_neurons, hidden_size=128).to(device)
mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

for epoch in range(n_epochs):
    mlp_model.train()
    for batch_x, batch_y in train_loader:
        mlp_optimizer.zero_grad()
        pred = mlp_model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        mlp_optimizer.step()

mlp_model.eval()
with torch.no_grad():
    mlp_predictions = mlp_model(X_test_t).cpu().numpy()

mlp_mse = mean_squared_error(Y_test, mlp_predictions)
mlp_r2 = r2_score(Y_test.flatten(), mlp_predictions.flatten())
print(f"\nMLP Results:")
print(f"  MSE: {mlp_mse:.4f}")
print(f"  R2:  {mlp_r2:.4f}")

# =============================================================================
# PART 5: Summary and Visualization
# =============================================================================

print("\n" + "="*60)
print("SUMMARY: Model Comparison")
print("="*60)

results = {
    'GLM (Poisson)': {'MSE': glm_mse, 'R2': glm_r2},
    'Neural ODE': {'MSE': node_mse, 'R2': node_r2},
    'MLP': {'MSE': mlp_mse, 'R2': mlp_r2}
}

print(f"\n{'Model':<20} {'MSE':<12} {'R2':<12}")
print("-" * 44)
for name, metrics in results.items():
    print(f"{name:<20} {metrics['MSE']:<12.4f} {metrics['R2']:<12.4f}")

# Best model
best_model = min(results.keys(), key=lambda k: results[k]['MSE'])
print(f"\nBest model (lowest MSE): {best_model}")

# Improvement over GLM
if results['Neural ODE']['MSE'] < results['GLM (Poisson)']['MSE']:
    improvement = (results['GLM (Poisson)']['MSE'] - results['Neural ODE']['MSE']) / results['GLM (Poisson)']['MSE'] * 100
    print(f"Neural ODE improvement over GLM: {improvement:.1f}%")
else:
    print("GLM performs better than Neural ODE on this data")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Training loss
axes[0].plot(losses)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Neural ODE Training Loss')

# Bar chart comparison
models = list(results.keys())
mse_values = [results[m]['MSE'] for m in models]
r2_values = [results[m]['R2'] for m in models]

axes[1].bar(models, mse_values, color=['blue', 'green', 'orange'])
axes[1].set_ylabel('MSE (lower is better)')
axes[1].set_title('Model Comparison - MSE')

axes[2].bar(models, r2_values, color=['blue', 'green', 'orange'])
axes[2].set_ylabel('R2 (higher is better)')
axes[2].set_title('Model Comparison - R2')

plt.tight_layout()
plt.savefig('neural_ode_vs_glm.png', dpi=150, bbox_inches='tight')
print("\nSaved: neural_ode_vs_glm.png")

print("\n" + "="*60)
print("DONE!")
print("="*60)
