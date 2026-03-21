"""
Simple LED Parameter Estimation for Quantum LED using Feed-Forward ANN
Using TensorFlow/Keras to predict PPFD values from (x, y, h) coordinates
Training with 13cm and 39cm height data
Testing with 31cm height data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
        return 0  # Default for unknown heights

def load_data():
    """Load measurement data from CSV files"""
    # Load coordinate mapping
    idx_points = pd.read_csv('idx_to_points.csv')
    idx_points.columns = idx_points.columns.str.strip().str.lower()

    # Load height data
    data_13cm = pd.read_csv('13_cm_led_height_ppfd_values.csv')
    data_31cm = pd.read_csv('27_cm_led_height_ppfd_values.csv')  # Test data (real=31cm)
    data_39cm = pd.read_csv('35_cm_led_height_ppfd_values.csv')  # Train data (real=39cm)

    # Clean column names
    data_13cm.columns = data_13cm.columns.str.strip().str.lower()
    data_31cm.columns = data_31cm.columns.str.strip().str.lower()
    data_39cm.columns = data_39cm.columns.str.strip().str.lower()

    return idx_points, data_13cm, data_31cm, data_39cm

def prepare_train_data(idx_points, data_13cm, data_39cm):
    """Prepare training data"""
    # Merge with coordinates
    df_13 = pd.merge(data_13cm, idx_points, on='idx', how='inner')
    df_39 = pd.merge(data_39cm, idx_points, on='idx', how='inner')

    # Extract coordinates and PPFD values
    x_13 = df_13['x'].values
    y_13 = df_13['y'].values
    ppfd_13 = df_13['quantum'].values * 0.8  # Real PPFD values

    x_39 = df_39['x'].values
    y_39 = df_39['y'].values
    ppfd_39 = df_39['quantum'].values * 0.8  # Real PPFD values

    # Real lamp heights (with dynamic offset)
    height_13 = 13 + get_height_offset(13)  # 13 cm
    height_39 = 35 + get_height_offset(35)  # 39 cm

    # Combine training data
    x_coords = np.concatenate([x_13, x_39])
    y_coords = np.concatenate([y_13, y_39])
    ppfd_measured = np.concatenate([ppfd_13, ppfd_39])
    lamp_heights = np.concatenate([
        np.full(len(x_13), height_13),
        np.full(len(x_39), height_39)
    ])

    # Create input features: [x, y, h]
    X = np.column_stack([x_coords, y_coords, lamp_heights])
    y = ppfd_measured

    return X, y, x_coords, y_coords, lamp_heights

def prepare_test_data(idx_points, data_31cm):
    """Prepare test data"""
    # Merge with coordinates
    df_31 = pd.merge(data_31cm, idx_points, on='idx', how='inner')

    # Extract coordinates and PPFD values
    x_31 = df_31['x'].values
    y_31 = df_31['y'].values
    ppfd_31 = df_31['quantum'].values * 0.8  # Real PPFD values

    # Real lamp height (with dynamic offset)
    height_31 = 27 + get_height_offset(27)  # 31 cm

    # Test data
    x_coords = x_31
    y_coords = y_31
    ppfd_measured = ppfd_31
    lamp_heights = np.full(len(x_31), height_31)

    # Create input features: [x, y, h]
    X = np.column_stack([x_coords, y_coords, lamp_heights])
    y = ppfd_measured

    return X, y, x_coords, y_coords, lamp_heights

def create_ann_model(input_dim=3):
    """Create simple feed-forward ANN model"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    return model

def calculate_metrics(y_true, y_pred):
    """Calculate R² and RMSE"""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / len(y_true))
    mae = np.mean(np.abs(residuals))
    return r_squared, rmse, mae

def plot_results(y_train, y_pred_train, y_test, y_pred_test, r2_train, rmse_train, r2_test, rmse_test):
    """Plot prediction results for both training and test data"""
    import os
    os.makedirs('results', exist_ok=True)

    fig = plt.figure(figsize=(16, 10))

    # Create grid for subplots
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ============= TRAINING DATA PLOTS =============
    # Plot 1: Training - Measured vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    measurement_points_train = np.arange(1, len(y_train) + 1)
    ax1.plot(measurement_points_train, y_train, 'o-', label='Measured',
             markersize=6, linewidth=1.5, color='blue')
    ax1.plot(measurement_points_train, y_pred_train, 's-', label='Predicted (ANN)',
             markersize=5, linewidth=1.5, color='red', alpha=0.7)
    ax1.set_xlabel('Measurement Point', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax1.set_title(f'TRAINING - Measured vs Predicted\nR²={r2_train:.4f}, RMSE={rmse_train:.2f}', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training - Scatter plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_train, y_pred_train, alpha=0.6, s=50, c='blue', edgecolor='black')
    min_val_train = min(y_train.min(), y_pred_train.min())
    max_val_train = max(y_train.max(), y_pred_train.max())
    ax2.plot([min_val_train, max_val_train], [min_val_train, max_val_train],
             'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Measured PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax2.set_title('TRAINING - Scatter Plot', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ============= TEST DATA PLOTS =============
    # Plot 3: Test - Measured vs Predicted
    ax3 = fig.add_subplot(gs[1, 0])
    measurement_points_test = np.arange(1, len(y_test) + 1)
    ax3.plot(measurement_points_test, y_test, 'o-', label='Measured',
             markersize=6, linewidth=1.5, color='green')
    ax3.plot(measurement_points_test, y_pred_test, 's-', label='Predicted (ANN)',
             markersize=5, linewidth=1.5, color='orange', alpha=0.7)
    ax3.set_xlabel('Measurement Point', fontsize=11, fontweight='bold')
    ax3.set_ylabel('PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax3.set_title(f'TEST (31cm) - Measured vs Predicted\nR²={r2_test:.4f}, RMSE={rmse_test:.2f}', fontsize=11, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Test - Scatter plot
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(y_test, y_pred_test, alpha=0.6, s=50, c='green', edgecolor='black')
    min_val_test = min(y_test.min(), y_pred_test.min())
    max_val_test = max(y_test.max(), y_pred_test.max())
    ax4.plot([min_val_test, max_val_test], [min_val_test, max_val_test],
             'r--', linewidth=2, label='Perfect Prediction')
    ax4.set_xlabel('Measured PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax4.set_title('TEST (31cm) - Scatter Plot', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Feed-Forward ANN - Quantum LED\nTraining: 13cm + 39cm  |  Test: 31cm (Interpolation)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('results/ann_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: results/ann_results.png")
    plt.close()

def main():
    """Main function"""
    print("=" * 60)
    print("Quantum LED Parameter Estimation using Feed-Forward ANN")
    print("Training: 13cm + 39cm | Testing: 31cm")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Step 1: Load data
    print("\nStep 1: Loading data...")
    idx_points, data_13cm, data_31cm, data_39cm = load_data()
    print(f"  - Loaded {len(idx_points)} coordinate points")
    print(f"  - Loaded 13cm (nominal) height data: {len(data_13cm)} points")
    print(f"  - Loaded 27cm (nominal->31cm real) height data: {len(data_31cm)} points")
    print(f"  - Loaded 35cm (nominal->39cm real) height data: {len(data_39cm)} points")

    # Step 2: Prepare training data
    print("\nStep 2: Preparing training data (13cm + 39cm)...")
    X_train, y_train, _, _, train_heights = prepare_train_data(idx_points, data_13cm, data_39cm)
    print(f"  - Total training points: {len(y_train)}")
    print(f"  - Input features: X (position), Y (position), H (lamp height)")
    print(f"  - PPFD range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"  - Heights: {np.unique(train_heights)} cm")

    # Step 3: Prepare test data
    print("\nStep 3: Preparing test data (31cm)...")
    X_test, y_test, _, _, test_heights = prepare_test_data(idx_points, data_31cm)
    print(f"  - Total test points: {len(y_test)}")
    print(f"  - PPFD range: [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"  - Heights: {np.unique(test_heights)} cm")

    # Step 4: Normalize data
    print("\nStep 4: Normalizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    X_test_scaled = scaler_X.transform(X_test)  # Use training scaler
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")

    # Step 5: Create model
    print("\nStep 5: Creating ANN model...")
    model = create_ann_model(input_dim=3)
    print(f"  - Architecture: 3 -> 64 -> 32 -> 16 -> 1")

    # Step 6: Compile model
    print("\nStep 6: Compiling model...")
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Step 7: Train model
    print("\nStep 7: Training model...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=1000,
        batch_size=4,
        verbose=1
    )

    # Step 8: Make predictions
    print("\nStep 8: Making predictions...")
    # Training predictions
    y_pred_train_scaled = model.predict(X_train_scaled, verbose=0).flatten()
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()

    # Test predictions
    y_pred_test_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()

    # Step 9: Calculate metrics
    print("\nStep 9: Calculating metrics...")
    r2_train, rmse_train, mae_train = calculate_metrics(y_train, y_pred_train)
    r2_test, rmse_test, mae_test = calculate_metrics(y_test, y_pred_test)

    print("\n  Results (Training Data - 13cm + 39cm):")
    print(f"    R²: {r2_train:.6f}")
    print(f"    RMSE: {rmse_train:.4f}")
    print(f"    MAE: {mae_train:.4f}")

    print("\n  Results (Test Data - 31cm):")
    print(f"    R²: {r2_test:.6f}")
    print(f"    RMSE: {rmse_test:.4f}")
    print(f"    MAE: {mae_test:.4f}")

    # Step 10: Plot results
    print("\nStep 10: Plotting results...")
    plot_results(y_train, y_pred_train, y_test, y_pred_test,
                 r2_train, rmse_train, r2_test, rmse_test)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Model: Simple Feed-Forward ANN")
    print(f"Architecture: 3 -> 64 -> 32 -> 16 -> 1")
    print(f"Training Data: {len(y_train)} samples (13cm + 39cm)")
    print(f"Test Data: {len(y_test)} samples (31cm)")
    print(f"\nPerformance (Training Data):")
    print(f"  R²: {r2_train:.6f}")
    print(f"  RMSE: {rmse_train:.4f} µmol/m²/s")
    print(f"  MAE: {mae_train:.4f} µmol/m²/s")
    print(f"\nPerformance (Test Data - 31cm):")
    print(f"  R²: {r2_test:.6f}")
    print(f"  RMSE: {rmse_test:.4f} µmol/m²/s")
    print(f"  MAE: {mae_test:.4f} µmol/m²/s")
    print("=" * 60)

    print("\nDone!")

if __name__ == "__main__":
    main()