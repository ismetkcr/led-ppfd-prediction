"""
Generate result graphs for the saved PINN model (Quantum LED).
Uses the exact same model architecture as new_led_visualization_gui.py.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.makedirs('results', exist_ok=True)


# ─── Exact GUI model definition ──────────────────────────────────────────────

def create_pinn_model(layer_sizes, activation_fn='relu'):
    """Exact replica of GUI's _create_pinn_model"""

    class FlexiblePINNModel(keras.Model):
        def __init__(self, layer_sizes, activation_fn):
            super().__init__()
            ffnn_layers = []
            for i, size in enumerate(layer_sizes):
                if i == 0:
                    ffnn_layers.append(layers.Dense(size, activation=activation_fn,
                                                    input_shape=(1,)))
                else:
                    ffnn_layers.append(layers.Dense(size, activation=activation_fn))
            ffnn_layers.append(layers.Dense(1, activation='linear'))
            self.ffnn = keras.Sequential(ffnn_layers, name='parameter_network')
            self.alpha_raw = tf.Variable(0.0, trainable=True, name='alpha_raw')

        def call(self, inputs, led_positions, training=False):
            x_points = inputs[:, 0:1]
            y_points = inputs[:, 1:2]
            h_points = inputs[:, 2:3]
            A_raw = self.ffnn(h_points, training=training)
            A = tf.nn.softplus(A_raw)
            alpha = 0.01 + 0.5 * tf.nn.sigmoid(self.alpha_raw)
            alpha = tf.broadcast_to(alpha, tf.shape(A))
            led_pos = tf.cast(led_positions, tf.float32)
            dx = x_points - led_pos[:, 0]
            dy = y_points - led_pos[:, 1]
            R = tf.sqrt(dx**2 + dy**2 + h_points**2 + 1e-8)
            h_over_R3 = h_points / (R**3 + 1e-8)
            exp_term = tf.exp(-alpha * R)
            return A * tf.reduce_sum(h_over_R3 * exp_term, axis=1, keepdims=True)

        def get_parameters(self, h_values):
            h_t = tf.constant(h_values.reshape(-1, 1), dtype=tf.float32)
            A_vals = tf.nn.softplus(self.ffnn(h_t, training=False)).numpy().flatten()
            alpha_val = (0.01 + 0.5 * tf.nn.sigmoid(self.alpha_raw)).numpy()
            return A_vals, np.full_like(A_vals, alpha_val)

    return FlexiblePINNModel(layer_sizes, activation_fn)


# ─── Manual weight loading via h5py ──────────────────────────────────────────

def load_weights_from_h5(model, h5_path, led_positions):
    """
    Load weights from h5 file using positional matching.
    Structure: parameter_network/dense_N/{kernel:0, bias:0}
               top_level_model_weights/alpha_raw:0
    """
    # Build model first
    dummy_input = np.zeros((1, 3), dtype=np.float32)
    dummy_led = np.array(led_positions[:2], dtype=np.float32)
    _ = model(dummy_input, led_positions=dummy_led)

    with h5py.File(h5_path, 'r') as f:
        # --- Load FFNN layers by position ---
        pn = f['parameter_network']
        # Sort layer group names by their numeric suffix (dense_8, dense_9, ...)
        layer_names = sorted(pn.keys(), key=lambda x: int(x.split('_')[-1]))

        for keras_layer, h5_name in zip(model.ffnn.layers, layer_names):
            grp = pn[h5_name]
            kernel = np.array(grp['kernel:0'])
            bias = np.array(grp['bias:0'])
            keras_layer.set_weights([kernel, bias])

        # --- Load alpha_raw ---
        alpha_val = float(np.array(f['top_level_model_weights']['alpha_raw:0']))
        model.alpha_raw.assign(alpha_val)
        alpha_sigmoid = 0.01 + 0.5 / (1 + np.exp(-alpha_val))
        print(f"  Loaded {len(layer_names)} FFNN layers.")
        print(f"  alpha_raw = {alpha_val:.6f}  ->  alpha = {alpha_sigmoid:.6f} cm-1")

    return model


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_height_offset(h):
    return 0 if h == 13 else 4


def load_csv_data():
    idx_pts = pd.read_csv('idx_to_points.csv')
    idx_pts.columns = idx_pts.columns.str.strip().str.lower()
    rows = []
    for fname, nominal in [('13_cm_led_height_ppfd_values.csv', 13),
                            ('27_cm_led_height_ppfd_values.csv', 27),
                            ('35_cm_led_height_ppfd_values.csv', 35)]:
        d = pd.read_csv(fname)
        d.columns = d.columns.str.strip().str.lower()
        df = pd.merge(d, idx_pts, on='idx', how='inner')
        real_h = nominal + get_height_offset(nominal)
        rows.append({'x': df['x'].values, 'y': df['y'].values,
                     'ppfd': df['quantum'].values * 0.8,
                     'h': np.full(len(df), float(real_h)),
                     'label': f'{nominal} cm (real={real_h} cm)',
                     'nominal': nominal, 'real_h': real_h})
    return rows


def metrics(y_true, y_pred):
    res = y_true - y_pred
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return r2, np.sqrt(ss_res / len(y_true)), np.mean(np.abs(res))


# ─── Main ─────────────────────────────────────────────────────────────────────

print("Loading metadata...")
with open('saved_models/pinn_model_Quantum_metadata.json') as f:
    meta = json.load(f)

led_positions = np.array(meta['led_positions'], dtype=np.float32)
layer_neurons = meta['layer_neurons']   # [64, 32, 8]
activation = meta['activation']          # 'relu'
print(f"  LED count: {len(led_positions)}")
print(f"  Layer neurons: {layer_neurons}, activation: {activation}")

print("Creating model...")
model = create_pinn_model(layer_neurons, activation)

print("Loading weights...")
model = load_weights_from_h5(model, 'saved_models/pinn_model_Quantum.h5', led_positions)

# ─── Load and evaluate data ───────────────────────────────────────────────────

print("Loading measurement data...")
height_data = load_csv_data()

results_list = []
print("\nEvaluation results:")
print("="*55)
for d in height_data:
    X = np.column_stack([d['x'], d['y'], d['h']]).astype(np.float32)
    pred = model(X, led_positions=led_positions, training=False).numpy().flatten()
    r2, rmse, mae = metrics(d['ppfd'], pred)
    results_list.append({**d, 'predicted': pred, 'r2': r2, 'rmse': rmse, 'mae': mae})
    print(f"  {d['label']:25s}  R2={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
print("="*55)

# ─── Plot 1: Measured vs Predicted (line) ─────────────────────────────────────

colors_pred = ['#E53935', '#43A047', '#FB8C00']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, res, col in zip(axes, results_list, colors_pred):
    pts = np.arange(1, len(res['ppfd']) + 1)
    ax.plot(pts, res['ppfd'], 'o-', label='Measured', color='#1565C0',
            linewidth=1.5, markersize=6)
    ax.plot(pts, res['predicted'], 's-', label='PINN Predicted', color=col,
            linewidth=1.5, markersize=5, alpha=0.85)
    ax.set_xlabel('Measurement Point', fontsize=11, fontweight='bold')
    ax.set_ylabel('PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax.set_title(f'{res["label"]}\nR²={res["r2"]:.4f}, RMSE={res["rmse"]:.2f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('PINN Model — Measured vs Predicted (Quantum LED, All Heights)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/pinn_measured_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/pinn_measured_vs_predicted.png")

# ─── Plot 2: Scatter ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, res, col in zip(axes, results_list, colors_pred):
    ax.scatter(res['ppfd'], res['predicted'], s=70, color=col, edgecolor='black', alpha=0.75)
    lo = min(res['ppfd'].min(), res['predicted'].min()) * 0.95
    hi = max(res['ppfd'].max(), res['predicted'].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect Fit')
    ax.set_xlabel('Measured PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax.set_title(f'{res["label"]}\nR²={res["r2"]:.4f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('PINN Model — Scatter: Predicted vs Measured (Quantum LED)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/pinn_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/pinn_scatter.png")

# ─── Plot 3: Learned parameters A(h) and alpha ────────────────────────────────

h_range = np.linspace(10, 45, 200)
A_vals, alpha_vals = model.get_parameters(h_range)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_hs = [(13, 'blue'), (31, 'green'), (39, 'red')]

ax = axes[0]
ax.plot(h_range, A_vals, '#1976D2', linewidth=2.5, label='A(h)')
for th, tc in train_hs:
    ax.axvline(x=th, linestyle='--', color=tc, alpha=0.6, label=f'h = {th} cm')
ax.set_xlabel('Lamp Height h (cm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Amplitude A', fontsize=12, fontweight='bold')
ax.set_title('Learned Amplitude A vs Height', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(h_range, alpha_vals, '#388E3C', linewidth=2.5, label='α (global constant)')
for th, tc in train_hs:
    ax.axvline(x=th, linestyle='--', color=tc, alpha=0.6, label=f'h = {th} cm')
ax.set_xlabel('Lamp Height h (cm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Decay Rate α (cm⁻¹)', fontsize=12, fontweight='bold')
ax.set_title('Learned Decay Rate α (Global)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('PINN Learned Physical Parameters — Quantum LED',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/pinn_learned_parameters.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/pinn_learned_parameters.png")

# ─── Plot 4: Model comparison summary ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
heights = [r['label'] for r in results_list]
r2_vals = [r['r2'] for r in results_list]
rmse_vals = [r['rmse'] for r in results_list]

x = np.arange(len(heights))
w = 0.35
bars1 = ax.bar(x - w/2, r2_vals, w, label='R²', color='#1976D2', alpha=0.85)
ax2 = ax.twinx()
bars2 = ax2.bar(x + w/2, rmse_vals, w, label='RMSE', color='#E53935', alpha=0.85)

ax.set_ylabel('R²', fontsize=12, fontweight='bold', color='#1976D2')
ax2.set_ylabel('RMSE (µmol/m²/s)', fontsize=12, fontweight='bold', color='#E53935')
ax.set_xticks(x)
ax.set_xticklabels(heights, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title('PINN Performance Summary — Quantum LED (All Heights)',
             fontsize=13, fontweight='bold')

for bar, val in zip(bars1, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1565C0')
for bar, val in zip(bars2, rmse_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#B71C1C')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='lower right')
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('results/pinn_performance_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/pinn_performance_summary.png")

print("\nAll PINN result graphs saved to results/")
