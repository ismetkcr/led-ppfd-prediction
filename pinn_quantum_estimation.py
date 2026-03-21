"""
Flexible Physics-Informed Neural Network (PINN) for Quantum LED Parameter Estimation

Key Feature: LED positions are NOT embedded in the model!
- Train once with any LED configuration
- Use with different orientations (perpendicular/parallel)
- Use with multi-lamp setups
- Use with different LED counts

Model Logic:
1. Input: (x, y, h) + LED positions (external)
2. FFNN: h -> [A, α]
3. Physics: PPFD = A × Σ[h/R³ × exp(-α·R)] where R = distance from each LED
4. Loss: MSE(PPFD_predicted, PPFD_measured)

Training: 13cm + 39cm
Testing: 31cm (interpolation test)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

# LED specifications for Quantum
LED_SPECS = {
    'num_leds': 120,
    'length': 100  # cm
}

def get_height_offset(height):
    """
    Get height offset for Quantum LED based on nominal height.
    13cm -> offset=0 (real=13cm)
    27cm -> offset=4 (real=31cm)
    35cm -> offset=4 (real=39cm)
    """
    if height == 13:
        return 0
    elif height in [27, 35]:
        return 4
    else:
        return 0

def load_data():
    """Load measurement data from CSV files"""
    idx_points = pd.read_csv('idx_to_points.csv')
    idx_points.columns = idx_points.columns.str.strip().str.lower()

    data_13cm = pd.read_csv('13_cm_led_height_ppfd_values.csv')
    data_27cm = pd.read_csv('27_cm_led_height_ppfd_values.csv')
    data_35cm = pd.read_csv('35_cm_led_height_ppfd_values.csv')

    data_13cm.columns = data_13cm.columns.str.strip().str.lower()
    data_27cm.columns = data_27cm.columns.str.strip().str.lower()
    data_35cm.columns = data_35cm.columns.str.strip().str.lower()

    return idx_points, data_13cm, data_27cm, data_35cm

def prepare_train_data(idx_points, data_13cm, data_39cm):
    """Prepare training data from 13cm and 39cm heights"""
    # Merge with coordinates
    df_13 = pd.merge(data_13cm, idx_points, on='idx', how='inner')
    df_39 = pd.merge(data_39cm, idx_points, on='idx', how='inner')

    # Extract data for 13cm
    x_13 = df_13['x'].values
    y_13 = df_13['y'].values
    ppfd_13 = df_13['quantum'].values * 0.8  # Real PPFD
    h_13 = np.full(len(x_13), 13 + get_height_offset(13))  # Real height = 13cm

    # Extract data for 39cm
    x_39 = df_39['x'].values
    y_39 = df_39['y'].values
    ppfd_39 = df_39['quantum'].values * 0.8  # Real PPFD
    h_39 = np.full(len(x_39), 35 + get_height_offset(35))  # Real height = 39cm

    # Combine
    x = np.concatenate([x_13, x_39])
    y = np.concatenate([y_13, y_39])
    h = np.concatenate([h_13, h_39])
    ppfd = np.concatenate([ppfd_13, ppfd_39])

    return x, y, h, ppfd

def prepare_test_data(idx_points, data_31cm):
    """Prepare test data from 31cm height"""
    df_31 = pd.merge(data_31cm, idx_points, on='idx', how='inner')

    x = df_31['x'].values
    y = df_31['y'].values
    ppfd = df_31['quantum'].values * 0.8  # Real PPFD
    h = np.full(len(x), 27 + get_height_offset(27))  # Real height = 31cm

    return x, y, h, ppfd

def create_led_positions(center_x=0, center_y=0, orientation='perpendicular',
                        num_leds=120, length=100):
    """
    Create LED positions for a single lamp

    Args:
        center_x, center_y: Lamp center position
        orientation: 'perpendicular' (along X) or 'parallel' (along Y)
        num_leds: Number of LEDs
        length: Lamp length in cm

    Returns:
        led_positions: (num_leds, 2) array of (x, y) positions
    """
    positions = np.linspace(-length/2, length/2, num_leds)

    if orientation == 'perpendicular':
        # LEDs along X axis
        x_leds = center_x + positions
        y_leds = center_y + np.zeros(num_leds)
    else:  # parallel
        # LEDs along Y axis
        x_leds = center_x + np.zeros(num_leds)
        y_leds = center_y + positions

    return np.column_stack([x_leds, y_leds])


class FlexiblePINNModel(keras.Model):
    """
    Flexible Physics-Informed Neural Network Model with Selective Normalization

    LED positions are NOT embedded - provided externally for each prediction!

    Architecture:
    1. FFNN: h_normalized -> [A_raw]
    2. Physics: PPFD = A × Σ[h/R³ × exp(-α·R)] (all RAW values)

    Normalization:
    - h is normalized for FFNN input
    - PPFD is normalized for loss calculation
    - x, y, LED positions, A are kept RAW for physics

    Usage:
        model = FlexiblePINNModel()
        ppfd = model(inputs=[x, y, h], led_positions=led_pos, scaler_h=scaler_h)
    """

    def __init__(self):
        super(FlexiblePINNModel, self).__init__()

        # FFNN: h_normalized -> [A_raw]
        self.ffnn = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(1,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(4, activation='relu'),
            layers.Dense(1, activation='linear')
        ], name='parameter_network')

        self.alpha_raw = tf.Variable(0.0, trainable=True, name='alpha_raw')

    def call(self, inputs, led_positions, scaler_h=None, training=False):
        """
        Forward pass with selective normalization

        Args:
            inputs: (batch_size, 3) tensor of [x, y, h] (RAW values)
            led_positions: (num_leds, 2) array/tensor of LED (x, y) positions (RAW)
            scaler_h: StandardScaler for h normalization (optional)
            training: bool

        Returns:
            ppfd_predicted: (batch_size, 1) tensor (RAW PPFD)
        """
        # Extract coordinates (RAW)
        x_points = inputs[:, 0:1]  # (batch_size, 1) RAW
        y_points = inputs[:, 1:2]  # (batch_size, 1) RAW
        h_points = inputs[:, 2:3]  # (batch_size, 1) RAW

        # Normalize h for FFNN input
        if scaler_h is not None:
            h_normalized = (h_points - scaler_h.mean_[0]) / scaler_h.scale_[0]
        else:
            h_normalized = h_points

        # FFNN: h_normalized -> [A_raw]
        A_raw = self.ffnn(h_normalized, training=training)
        A = tf.nn.softplus(A_raw)  # A is RAW (not normalized)

        # α global (RAW decay rate)
        alpha = 0.01 + 0.5 * tf.nn.sigmoid(self.alpha_raw)
        alpha = tf.broadcast_to(alpha, tf.shape(A))

        # Physics equation with RAW values: PPFD = A × Σ[h/R³ × exp(-α·R)]
        ppfd = self._physics_model(x_points, y_points, h_points, A, alpha, led_positions)

        return ppfd

    def _physics_model(self, x_points, y_points, h_points, A, alpha, led_positions):
        """
        Physics model: PPFD = A × Σ[h/R³ × exp(-α·R)]

        Args:
            x_points: (batch_size, 1)
            y_points: (batch_size, 1)
            h_points: (batch_size, 1)
            A: (batch_size, 1)
            alpha: (batch_size, 1)
            led_positions: (num_leds, 2) - EXTERNAL LED positions

        Returns:
            ppfd: (batch_size, 1)
        """
        # Convert LED positions to tensors
        led_positions = tf.cast(led_positions, tf.float32)
        x_leds = led_positions[:, 0]  # (num_leds,)
        y_leds = led_positions[:, 1]  # (num_leds,)

        # Calculate distances from all LEDs to all points
        # Broadcasting: (batch_size, 1) - (num_leds,) -> (batch_size, num_leds)
        dx = x_points - x_leds  # (batch_size, num_leds)
        dy = y_points - y_leds  # (batch_size, num_leds)
        dz = -h_points          # LEDs at z=h, measurement at z=0

        # R = sqrt(dx² + dy² + dz²)
        R = tf.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)  # Add epsilon for stability

        # Physics equation per LED:
        # contribution_i = h / R_i³ × exp(-α × R_i)

        # h / R³ -> (batch_size, 1) / (batch_size, num_leds)
        h_over_R3 = h_points / (R ** 3 + 1e-8)  # (batch_size, num_leds)

        # exp(-α × R) -> (batch_size, 1) × (batch_size, num_leds)
        exp_term = tf.exp(-alpha * R)  # (batch_size, num_leds)

        # Sum over all LEDs: Σ[h/R³ × exp(-α·R)]
        sum_contribution = tf.reduce_sum(h_over_R3 * exp_term, axis=1, keepdims=True)

        # PPFD = A × Σ
        ppfd = A * sum_contribution  # (batch_size, 1)

        return ppfd

    def get_parameters(self, h_values, scaler_h=None):
        """
        Get parameters A and α for given height values

        Args:
            h_values: array of height values (RAW)
            scaler_h: StandardScaler for h normalization (optional)
        """
        h_tensor = tf.constant(h_values.reshape(-1, 1), dtype=tf.float32)

        # Normalize h if scaler provided
        if scaler_h is not None:
            h_normalized = (h_tensor - scaler_h.mean_[0]) / scaler_h.scale_[0]
        else:
            h_normalized = h_tensor

        A_raw = self.ffnn(h_normalized, training=False)
        A_values = tf.nn.softplus(A_raw).numpy().flatten()

        # α global value
        alpha_value = (0.01 + 0.5 * tf.nn.sigmoid(self.alpha_raw)).numpy()
        alpha_values = np.full_like(A_values, alpha_value)

        return A_values, alpha_values


def train_model(model, x_train, y_train, h_train, ppfd_train, led_positions,
                epochs=1000, batch_size=32, learning_rate=0.001):
    """
    Train flexible PINN model with selective normalization

    Args:
        model: FlexiblePINNModel instance
        x_train, y_train, h_train: coordinate arrays (RAW)
        ppfd_train: measured PPFD values (RAW)
        led_positions: LED positions (RAW)
        epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate

    Returns:
        history: dict with training metrics
        scaler_h: StandardScaler for h only
        scaler_ppfd: StandardScaler for PPFD only
    """
    # Prepare dataset (RAW values)
    X_train = np.column_stack([x_train, y_train, h_train]).astype(np.float32)
    ppfd_train_reshaped = ppfd_train.reshape(-1, 1).astype(np.float32)

    # Normalize ONLY h (for FFNN input)
    scaler_h = StandardScaler()
    h_train_reshaped = h_train.reshape(-1, 1)
    scaler_h.fit(h_train_reshaped)

    # Normalize ONLY PPFD (for loss calculation)
    scaler_ppfd = StandardScaler()
    ppfd_train_normalized = scaler_ppfd.fit_transform(ppfd_train_reshaped).astype(np.float32)

    print("\n" + "="*60)
    print("SELECTIVE NORMALIZATION (Option 2)")
    print("="*60)
    print(f"✓ h normalized: mean={scaler_h.mean_[0]:.2f}, std={scaler_h.scale_[0]:.2f}")
    print(f"✓ PPFD normalized: mean={scaler_ppfd.mean_[0]:.2f}, std={scaler_ppfd.scale_[0]:.2f}")
    print(f"✗ x, y kept RAW (for physics)")
    print(f"✗ LED positions kept RAW (for physics): {len(led_positions)} LEDs")
    print(f"✗ A kept RAW (output of FFNN)")
    print("="*60)

    # Dataset with RAW inputs and normalized PPFD
    dataset = tf.data.Dataset.from_tensor_slices((X_train, ppfd_train_normalized))
    dataset = dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Loss function
    loss_fn = keras.losses.MeanSquaredError()

    # Training loop
    history = {'loss': [], 'mae': []}

    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LED positions (RAW): {led_positions.shape[0]} LEDs")
    print("-"*60)

    for epoch in range(epochs):
        epoch_loss = []
        epoch_mae = []

        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                # Forward pass: RAW inputs, RAW LED positions, scaler_h for h normalization inside model
                # Model returns RAW PPFD
                predictions_raw = model(batch_x, led_positions=led_positions,
                                      scaler_h=scaler_h, training=True)

                # Normalize predictions for loss calculation
                predictions_normalized = (predictions_raw - scaler_ppfd.mean_[0]) / scaler_ppfd.scale_[0]

                # Calculate loss with normalized values
                loss = loss_fn(batch_y, predictions_normalized)

            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Track metrics (in normalized space)
            epoch_loss.append(loss.numpy())
            mae = tf.reduce_mean(tf.abs(batch_y - predictions_normalized)).numpy()
            epoch_mae.append(mae)

        # Average metrics for epoch
        avg_loss = np.mean(epoch_loss)
        avg_mae = np.mean(epoch_mae)

        history['loss'].append(avg_loss)
        history['mae'].append(avg_mae)

        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.6f} | MAE: {avg_mae:.4f}")

    print("-"*60)
    print("TRAINING COMPLETED")
    print("="*60)

    return history, scaler_h, scaler_ppfd


def evaluate_model(model, x, y, h, ppfd_measured, led_positions, scaler_h, scaler_ppfd):
    """Evaluate model performance with selective normalization"""
    # Prepare input (RAW values)
    X = np.column_stack([x, y, h]).astype(np.float32)

    # Predict: Model takes RAW inputs, RAW LED positions, scaler_h for h normalization
    # Model returns RAW PPFD directly
    ppfd_pred = model(X, led_positions=led_positions,
                     scaler_h=scaler_h, training=False).numpy().flatten()

    # Calculate metrics
    residuals = ppfd_measured - ppfd_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ppfd_measured - ppfd_measured.mean())**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / len(ppfd_measured))
    mae = np.mean(np.abs(residuals))

    return ppfd_pred, r2, rmse, mae


def plot_training_history(history):
    """Plot training loss and MAE"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
    axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # MAE
    axes[1].plot(epochs, history['mae'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('MAE', fontsize=11, fontweight='bold')
    axes[1].set_title('Training MAE', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_results(ppfd_train, ppfd_pred_train, ppfd_test, ppfd_pred_test,
                r2_train, rmse_train, r2_test, rmse_test):
    """Plot prediction results"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Training - Line plot
    ax1 = fig.add_subplot(gs[0, 0])
    points = np.arange(1, len(ppfd_train) + 1)
    ax1.plot(points, ppfd_train, 'o-', label='Measured', markersize=5, linewidth=1.5, color='blue')
    ax1.plot(points, ppfd_pred_train, 's-', label='PINN', markersize=4, linewidth=1.5, color='red', alpha=0.7)
    ax1.set_xlabel('Measurement Point', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax1.set_title(f'TRAINING - R²={r2_train:.4f}, RMSE={rmse_train:.2f}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Training - Scatter
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(ppfd_train, ppfd_pred_train, alpha=0.6, s=50, c='blue', edgecolor='black')
    min_v = min(ppfd_train.min(), ppfd_pred_train.min())
    max_v = max(ppfd_train.max(), ppfd_pred_train.max())
    ax2.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Measured PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax2.set_title('TRAINING - Scatter Plot', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Test - Line plot
    ax3 = fig.add_subplot(gs[1, 0])
    points = np.arange(1, len(ppfd_test) + 1)
    ax3.plot(points, ppfd_test, 'o-', label='Measured', markersize=5, linewidth=1.5, color='green')
    ax3.plot(points, ppfd_pred_test, 's-', label='PINN', markersize=4, linewidth=1.5, color='orange', alpha=0.7)
    ax3.set_xlabel('Measurement Point', fontsize=11, fontweight='bold')
    ax3.set_ylabel('PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax3.set_title(f'TEST (31cm) - R²={r2_test:.4f}, RMSE={rmse_test:.2f}', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Test - Scatter
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(ppfd_test, ppfd_pred_test, alpha=0.6, s=50, c='green', edgecolor='black')
    min_v = min(ppfd_test.min(), ppfd_pred_test.min())
    max_v = max(ppfd_test.max(), ppfd_pred_test.max())
    ax4.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect Prediction')
    ax4.set_xlabel('Measured PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax4.set_title('TEST (31cm) - Scatter Plot', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Flexible PINN Results: Training (13+39cm) vs Test (31cm)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.show()


def plot_learned_parameters(model, scaler_h):
    """Plot learned A and α parameters across heights"""
    h_range = np.linspace(10, 45, 100)
    A_values, alpha_values = model.get_parameters(h_range, scaler_h)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # A vs height
    axes[0].plot(h_range, A_values, 'b-', linewidth=2)
    axes[0].axvline(x=13, color='r', linestyle='--', alpha=0.5, label='Training: 13cm')
    axes[0].axvline(x=39, color='r', linestyle='--', alpha=0.5, label='Training: 39cm')
    axes[0].axvline(x=31, color='g', linestyle='--', alpha=0.5, label='Test: 31cm')
    axes[0].set_xlabel('Height (cm)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('A (Intensity Parameter)', fontsize=11, fontweight='bold')
    axes[0].set_title('Learned A vs Height', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # α vs height
    axes[1].plot(h_range, alpha_values, 'b-', linewidth=2)
    axes[1].axvline(x=13, color='r', linestyle='--', alpha=0.5, label='Training: 13cm')
    axes[1].axvline(x=39, color='r', linestyle='--', alpha=0.5, label='Training: 39cm')
    axes[1].axvline(x=31, color='g', linestyle='--', alpha=0.5, label='Test: 31cm')
    axes[1].set_xlabel('Height (cm)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('α (Decay Rate, cm⁻¹)', fontsize=11, fontweight='bold')
    axes[1].set_title('Learned α vs Height', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()




def main():
    """Main function"""
    print("=" * 60)
    print("FLEXIBLE Physics-Informed Neural Network (PINN)")
    print("Quantum LED Parameter Estimation")
    print("=" * 60)
    print("Model: h -> FFNN -> [A, α] -> Physics(LED_pos) -> PPFD")
    print("Physics: PPFD = A × Σ[h/R³ × exp(-α·R)]")
    print("Key Feature: LED positions NOT embedded in model!")
    print("Training: 13cm + 39cm")
    print("Testing: 31cm (interpolation)")
    print("=" * 60)

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Step 1: Load data
    print("\nStep 1: Loading data...")
    idx_points, data_13cm, data_27cm, data_35cm = load_data()
    print(f"  Loaded coordinate points: {len(idx_points)}")

    # Step 2: Prepare data
    print("\nStep 2: Preparing data...")
    x_train, y_train, h_train, ppfd_train = prepare_train_data(idx_points, data_13cm, data_35cm)
    x_test, y_test, h_test, ppfd_test = prepare_test_data(idx_points, data_27cm)
    print(f"  Training: {len(ppfd_train)} points at heights {np.unique(h_train)} cm")
    print(f"  Testing: {len(ppfd_test)} points at height {np.unique(h_test)} cm")

    # Step 3: Create LED positions (for training)
    print("\nStep 3: Creating LED positions for training...")
    led_positions_train = create_led_positions(center_x=0, center_y=0, orientation='perpendicular')
    print(f"  Training LED config: Perpendicular, center=(0,0)")
    print(f"  LEDs: {led_positions_train.shape[0]}")

    # Step 4: Create flexible model
    print("\nStep 4: Creating Flexible PINN model...")
    model = FlexiblePINNModel()
    print(f"  FFNN architecture: 1 -> 32 -> 16 -> 8 -> 2")
    print(f"  Output: [A, α] (height-dependent parameters)")
    print(f"  LED positions: Provided externally at prediction time!")

    # Step 5: Train model
    print("\nStep 5: Training model with selective normalization (Option 2)...")
    history, scaler_h, scaler_ppfd = train_model(
        model, x_train, y_train, h_train, ppfd_train,
        led_positions=led_positions_train,
        epochs=5000,
        batch_size=4,
        learning_rate=0.0005
    )

    # Step 6: Plot training history
    print("\nStep 6: Plotting training history...")
    plot_training_history(history)

    # Step 7: Evaluate on training data
    print("\nStep 7: Evaluating on training data...")
    ppfd_pred_train, r2_train, rmse_train, mae_train = evaluate_model(
        model, x_train, y_train, h_train, ppfd_train, led_positions_train,
        scaler_h, scaler_ppfd)
    print(f"  Training Results:")
    print(f"    R²: {r2_train:.6f}")
    print(f"    RMSE: {rmse_train:.4f} µmol/m²/s")
    print(f"    MAE: {mae_train:.4f} µmol/m²/s")

    # Step 8: Evaluate on test data (SAME LED config)
    print("\nStep 8: Evaluating on test data (31cm - interpolation)...")
    ppfd_pred_test, r2_test, rmse_test, mae_test = evaluate_model(
        model, x_test, y_test, h_test, ppfd_test, led_positions_train,
        scaler_h, scaler_ppfd)
    print(f"  Test Results:")
    print(f"    R²: {r2_test:.6f}")
    print(f"    RMSE: {rmse_test:.4f} µmol/m²/s")
    print(f"    MAE: {mae_test:.4f} µmol/m²/s")

    # Step 9: Plot results
    print("\nStep 9: Plotting prediction results...")
    plot_results(ppfd_train, ppfd_pred_train, ppfd_test, ppfd_pred_test,
                r2_train, rmse_train, r2_test, rmse_test)

    # Step 10: Plot learned parameters
    print("\nStep 10: Plotting learned parameters (A and α)...")
    plot_learned_parameters(model, scaler_h)

     

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Model: Flexible PINN (LED positions external)")
    print(f"Architecture: h -> FFNN(32->16->8->2) -> [A, α]")
    print(f"Physics: PPFD = A × Σ[h/R³ × exp(-α·R)]")
    print(f"\nTraining Data: {len(ppfd_train)} points (13cm + 39cm)")
    print(f"  R² = {r2_train:.6f}")
    print(f"  RMSE = {rmse_train:.4f} µmol/m²/s")
    print(f"\nTest Data: {len(ppfd_test)} points (31cm - interpolation)")
    print(f"  R² = {r2_test:.6f}")
    print(f"  RMSE = {rmse_test:.4f} µmol/m²/s")
    print("\n✅ Model can now be used with:")
    print("   - Different LED orientations (perpendicular/parallel)")
    print("   - Multiple lamps (superposition)")
    print("   - Different LED configurations")
    print("   - Any number of LEDs")
    print("=" * 60)
    print("\nDone!")

if __name__ == "__main__":
    main()
