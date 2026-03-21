"""
Simple LED Parameter Estimation for Quantum LED
Using 17cm and 31cm height data with Grid Search and Golden Section Search
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar

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
    data_17cm = pd.read_csv('13_cm_led_height_ppfd_values.csv')
    data_31cm = pd.read_csv('35_cm_led_height_ppfd_values.csv')

    # Clean column names
    data_17cm.columns = data_17cm.columns.str.strip().str.lower()
    data_31cm.columns = data_31cm.columns.str.strip().str.lower()

    return idx_points, data_17cm, data_31cm

def prepare_data(idx_points, data_17cm, data_31cm):
    """Prepare data for estimation"""
    # Merge with coordinates
    df_17 = pd.merge(data_17cm, idx_points, on='idx', how='inner')
    df_31 = pd.merge(data_31cm, idx_points, on='idx', how='inner')

    # Extract coordinates and PPFD values
    x_17 = df_17['x'].values
    y_17 = df_17['y'].values
    ppfd_17 = df_17['quantum'].values * 0.8  # Real PPFD values

    x_31 = df_31['x'].values
    y_31 = df_31['y'].values
    ppfd_31 = df_31['quantum'].values * 0.8  # Real PPFD values

    # Real lamp heights (with dynamic offset)
    # Data files: 13_cm and 27_cm (nominal heights)
    nominal_height_1 = 13  # First data file
    nominal_height_2 = 27  # Second data file
    height_13 = nominal_height_1 + get_height_offset(nominal_height_1)  # 13 + 0 = 13 cm
    height_27 = nominal_height_2 + get_height_offset(nominal_height_2)  # 27 + 4 = 31 cm

    # Combine data
    x_coords = np.concatenate([x_17, x_31])
    y_coords = np.concatenate([y_17, y_31])
    ppfd_measured = np.concatenate([ppfd_17, ppfd_31])
    lamp_heights = np.concatenate([
        np.full(len(x_17), height_13),
        np.full(len(x_31), height_27)
    ])

    return x_coords, y_coords, ppfd_measured, lamp_heights

def create_led_positions():
    """Create LED positions (perpendicular orientation)"""
    num_leds = LED_SPECS['num_leds']
    length = LED_SPECS['length']

    x_leds = np.linspace(-length/2, length/2, num_leds)
    y_leds = np.zeros(num_leds)

    return x_leds, y_leds

def calculate_distance_matrix(x_coords, y_coords, lamp_heights, x_leds, y_leds):
    """Calculate distance matrix between LEDs and measurement points"""
    num_leds = len(x_leds)
    num_points = len(x_coords)

    R_matrix = np.zeros((num_leds, num_points))

    for j in range(num_points):
        z_lamp = lamp_heights[j]
        for i in range(num_leds):
            dx = x_coords[j] - x_leds[i]
            dy = y_coords[j] - y_leds[i]
            dz = 0 - z_lamp  # Measurement at z=0, LEDs at z=lamp_height
            R_matrix[i, j] = np.sqrt(dx**2 + dy**2 + dz**2)

    return R_matrix

def calculate_optimal_theta1(alpha, R_matrix, ppfd_measured):
    """Calculate optimal theta1 for given alpha"""
    S = np.sum(np.exp(-alpha * R_matrix), axis=0)
    numerator = np.dot(S, ppfd_measured)
    denominator = np.dot(S, S)
    theta1_opt = numerator / denominator if denominator > 0 else 0
    return theta1_opt, S

def calculate_sse(alpha, R_matrix, ppfd_measured):
    """Calculate Sum of Squared Errors for given alpha"""
    theta1_opt, S = calculate_optimal_theta1(alpha, R_matrix, ppfd_measured)
    ppfd_predicted = theta1_opt * S
    residuals = ppfd_measured - ppfd_predicted
    sse = np.sum(residuals**2)
    return sse, theta1_opt, ppfd_predicted

def grid_search(R_matrix, ppfd_measured, alpha_min=0.001, alpha_max=0.5, n_alpha=500):
    """Perform grid search to find optimal alpha"""
    print("\n=== Grid Search ===")
    print(f"Alpha range: [{alpha_min}, {alpha_max}]")
    print(f"Number of points: {n_alpha}")

    alpha_values = np.linspace(alpha_min, alpha_max, n_alpha)
    sse_values = np.zeros(n_alpha)
    theta1_values = np.zeros(n_alpha)

    for i, alpha in enumerate(alpha_values):
        sse, theta1, _ = calculate_sse(alpha, R_matrix, ppfd_measured)
        sse_values[i] = sse
        theta1_values[i] = theta1

    # Find minimum
    min_idx = np.argmin(sse_values)
    alpha_opt = alpha_values[min_idx]
    theta1_opt = theta1_values[min_idx]
    sse_opt = sse_values[min_idx]

    print(f"Optimal alpha: {alpha_opt:.8f}")
    print(f"Optimal theta1: {theta1_opt:.8f}")
    print(f"SSE: {sse_opt:.4f}")

    return alpha_opt, theta1_opt, sse_opt, alpha_values, sse_values

def golden_section_search(R_matrix, ppfd_measured, alpha_max=0.5):
    """Perform golden section search to find optimal alpha"""
    print("\n=== Golden Section Search ===")
    print(f"Alpha range: [0, {alpha_max}]")

    def cost_func(alpha):
        sse, _, _ = calculate_sse(alpha, R_matrix, ppfd_measured)
        return sse

    result = minimize_scalar(cost_func, bounds=(0, alpha_max), method='bounded',
                            options={'xatol': 1e-12})

    alpha_opt = result.x
    sse_opt = result.fun
    theta1_opt, _ = calculate_optimal_theta1(alpha_opt, R_matrix, ppfd_measured)

    print(f"Optimal alpha: {alpha_opt:.8f}")
    print(f"Optimal theta1: {theta1_opt:.8f}")
    print(f"SSE: {sse_opt:.4f}")

    return alpha_opt, theta1_opt, sse_opt

def calculate_metrics(ppfd_measured, ppfd_predicted):
    """Calculate R² and RMSE"""
    residuals = ppfd_measured - ppfd_predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ppfd_measured - ppfd_measured.mean())**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / len(ppfd_measured))
    return r_squared, rmse

def plot_results(ppfd_measured, ppfd_predicted_grid, ppfd_predicted_golden,
                alpha_values, sse_values, r2_grid, rmse_grid, r2_golden, rmse_golden,
                title_suffix=""):
    """Plot measurement points and predicted values"""
    import os
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Measurement points (1 to n)
    measurement_points = np.arange(1, len(ppfd_measured) + 1)

    # Plot 1: Grid Search Results
    ax1 = axes[0]
    ax1.plot(measurement_points, ppfd_measured, 'o-', label='Measured',
             markersize=6, linewidth=1.5, color='blue')
    ax1.plot(measurement_points, ppfd_predicted_grid, 's-', label='Predicted (Grid)',
             markersize=5, linewidth=1.5, color='red', alpha=0.7)
    ax1.set_xlabel('Measurement Point', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Grid Search\nR²={r2_grid:.4f}, RMSE={rmse_grid:.2f}', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Golden Section Search Results
    ax2 = axes[1]
    ax2.plot(measurement_points, ppfd_measured, 'o-', label='Measured',
             markersize=6, linewidth=1.5, color='blue')
    ax2.plot(measurement_points, ppfd_predicted_golden, '^-', label='Predicted (Golden)',
             markersize=5, linewidth=1.5, color='green', alpha=0.7)
    ax2.set_xlabel('Measurement Point', fontsize=11, fontweight='bold')
    ax2.set_ylabel('PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Golden Section Search\nR²={r2_golden:.4f}, RMSE={rmse_golden:.2f}', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Grid Search SSE vs Alpha
    ax3 = axes[2]
    ax3.plot(alpha_values, sse_values, 'b-', linewidth=2)
    ax3.set_xlabel('Alpha (α)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('SSE', fontsize=11, fontweight='bold')
    ax3.set_title('Grid Search - SSE vs Alpha', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Linear Regression (Exponential Decay) - Quantum LED', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = f'results/linear_regression_results{title_suffix}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: {save_path}")
    plt.close()

def main():
    """Main function"""
    print("=" * 60)
    print("Quantum LED Parameter Estimation")
    print("Using 13cm (real=13cm) and 27cm (real=31cm) height data")
    print("=" * 60)

    # Step 1: Load data
    print("\nStep 1: Loading data...")
    idx_points, data_17cm, data_31cm = load_data()
    print(f"  - Loaded {len(idx_points)} coordinate points")
    print(f"  - Loaded 13cm (nominal) height data: {len(data_17cm)} points")
    print(f"  - Loaded 27cm (nominal) height data: {len(data_31cm)} points")

    # Step 2: Prepare data
    print("\nStep 2: Preparing data...")
    x_coords, y_coords, ppfd_measured, lamp_heights = prepare_data(idx_points, data_17cm, data_31cm)
    print(f"  - Total measurement points: {len(ppfd_measured)}")
    print(f"  - PPFD range: [{ppfd_measured.min():.2f}, {ppfd_measured.max():.2f}]")
    print(f"  - Heights: {np.unique(lamp_heights)} cm")

    # Step 3: Create LED positions
    print("\nStep 3: Creating LED positions...")
    x_leds, y_leds = create_led_positions()
    print(f"  - Number of LEDs: {len(x_leds)}")
    print(f"  - LED arrangement: Perpendicular (along X-axis)")

    # Step 4: Calculate distance matrix
    print("\nStep 4: Calculating distance matrix...")
    R_matrix = calculate_distance_matrix(x_coords, y_coords, lamp_heights, x_leds, y_leds)
    print(f"  - Distance matrix shape: {R_matrix.shape}")

    # Step 5: Grid Search
    alpha_grid, theta1_grid, sse_grid, alpha_values, sse_values = grid_search(
        R_matrix, ppfd_measured, alpha_min=0.001, alpha_max=1, n_alpha=500
    )
    _, S_grid = calculate_optimal_theta1(alpha_grid, R_matrix, ppfd_measured)
    ppfd_predicted_grid = theta1_grid * S_grid
    r2_grid, rmse_grid = calculate_metrics(ppfd_measured, ppfd_predicted_grid)
    print(f"R²: {r2_grid:.6f}")
    print(f"RMSE: {rmse_grid:.4f}")

    # Step 6: Golden Section Search
    alpha_golden, theta1_golden, sse_golden = golden_section_search(
        R_matrix, ppfd_measured, alpha_max=1
    )
    _, S_golden = calculate_optimal_theta1(alpha_golden, R_matrix, ppfd_measured)
    ppfd_predicted_golden = theta1_golden * S_golden
    r2_golden, rmse_golden = calculate_metrics(ppfd_measured, ppfd_predicted_golden)
    print(f"R²: {r2_golden:.6f}")
    print(f"RMSE: {rmse_golden:.4f}")

    # Step 7: Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Method':<20} {'Alpha':<15} {'Theta1':<15} {'R²':<12} {'RMSE':<10}")
    print("-" * 60)
    print(f"{'Grid Search':<20} {alpha_grid:<15.8f} {theta1_grid:<15.8f} {r2_grid:<12.6f} {rmse_grid:<10.4f}")
    print(f"{'Golden Section':<20} {alpha_golden:<15.8f} {theta1_golden:<15.8f} {r2_golden:<12.6f} {rmse_golden:<10.4f}")
    print("=" * 60)

    # Step 8: Plot results
    print("\nStep 8: Creating visualization...")
    plot_results(ppfd_measured, ppfd_predicted_grid, ppfd_predicted_golden,
                alpha_values, sse_values, r2_grid, rmse_grid, r2_golden, rmse_golden)

    print("\nDone!")

if __name__ == "__main__":
    main()
