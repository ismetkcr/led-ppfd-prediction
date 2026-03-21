"""
LED Multi-Height Visualization GUI
Visualizes PPFD measurements for 3 LED types across different lamp heights
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import os
from pathlib import Path
from scipy.optimize import minimize_scalar
from datetime import datetime


class LEDVisualizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LED Multi-Height Visualization & Parameter Estimation")
        self.root.geometry("1600x900")

        # Data storage
        self.idx_points_df = None
        self.lamp_data = {}  # {height: dataframe}
        self.available_heights = []

        # LED specifications
        self.led_specs = {
            'Quantum': {'num_leds': 120, 'length': 100, 'cost': 2200},
            '12V': {'num_leds': 144, 'length': 100, 'cost': 550},
            '54V': {'num_leds': 72, 'length': 100, 'cost': 850}
        }

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 1: Visualization
        self.visualization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.visualization_frame, text='Visualization')

        # Tab 2: Linear Regression Estimation
        self.estimation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.estimation_frame, text='Linear Regression Estimation')

        # Tab 3: Feed-Forward ANN Estimation
        self.ann_estimation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ann_estimation_frame, text='Feed-Forward ANN Estimation')

        # Tab 4: PINN Estimation
        self.pinn_estimation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pinn_estimation_frame, text='PINN Estimation')

        # Tab 5: Custom LED Layout
        self.custom_layout_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.custom_layout_frame, text='Custom LED Layout')

        # Create interfaces for all tabs
        self.create_visualization_interface()
        self.create_estimation_interface()
        self.create_ann_estimation_interface()
        self.create_pinn_estimation_interface()
        self.create_custom_layout_interface()

        # Auto-load data on startup
        self.load_data_from_program_directory()

    def get_height_offset(self, led_type, height):
        """
        Get height offset for a given LED type and nominal height.
        For Quantum LED: 13cm -> offset=0, 27cm and 35cm -> offset=4
        For other LEDs: offset=0
        """
        if led_type == 'Quantum':
            if height == 13:
                return 0
            elif height in [27, 35]:
                return 4
            else:
                return 0  # Default for unknown heights
        else:
            return 0

    def create_visualization_interface(self):
        """Create the visualization tab interface"""

        # Top control panel
        control_frame = ttk.Frame(self.visualization_frame)
        control_frame.pack(side='top', fill='x', padx=10, pady=10)

        # Title
        title_label = ttk.Label(control_frame, text="LED PPFD Visualization - Multi-Height Analysis",
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)

        # File loading status
        file_frame = ttk.LabelFrame(control_frame, text="Data Status", padding=10)
        file_frame.pack(fill='x', pady=5)

        self.data_status_label = ttk.Label(file_frame, text="Loading data...",
                                          font=('Arial', 10, 'italic'), foreground='orange')
        self.data_status_label.pack(side='left', padx=20)

        # Selection frame
        selection_frame = ttk.LabelFrame(control_frame, text="LED Type Selection", padding=10)
        selection_frame.pack(fill='x', pady=5)

        # LED type selection
        ttk.Label(selection_frame, text="Select LED Type:",
                 font=('Arial', 11, 'bold')).pack(side='left', padx=5)

        self.led_type_var = tk.StringVar(value='Quantum')
        led_combo = ttk.Combobox(selection_frame, textvariable=self.led_type_var,
                                values=['Quantum', '12V', '54V'],
                                state='readonly', width=15)
        led_combo.pack(side='left', padx=5)

        # Buttons
        ttk.Button(selection_frame, text="Visualize All Heights",
                  command=self.visualize_all_heights,
                  width=20).pack(side='left', padx=10)

        ttk.Button(selection_frame, text="Clear",
                  command=self.clear_visualization,
                  width=10).pack(side='left', padx=5)

        # Main content area (will contain graphs and table)
        self.content_frame = ttk.Frame(self.visualization_frame)
        self.content_frame.pack(side='top', fill='both', expand=True, padx=10, pady=10)

    def load_data_from_program_directory(self):
        """Load all CSV files from the same directory as the program"""
        # Get the directory where the program is located
        directory = os.path.dirname(os.path.abspath(__file__))

        try:
            # Find idx_to_points.csv
            idx_file = None
            for file in os.listdir(directory):
                if 'idx_to_points' in file.lower() and file.endswith('.csv'):
                    idx_file = os.path.join(directory, file)
                    break

            if not idx_file:
                messagebox.showerror("Error", "Could not find 'idx_to_points.csv' file!")
                return

            # Load idx_to_points
            self.idx_points_df = pd.read_csv(idx_file)

            # Normalize column names (remove spaces, convert to lowercase)
            self.idx_points_df.columns = self.idx_points_df.columns.str.strip().str.lower()

            # Validate idx_to_points structure
            if not all(col in self.idx_points_df.columns for col in ['idx', 'x', 'y']):
                messagebox.showerror("Error", f"idx_to_points.csv must contain columns: idx, X, Y\nFound: {list(self.idx_points_df.columns)}")
                return

            # Rename for consistency
            self.idx_points_df = self.idx_points_df.rename(columns={'idx': 'idx', 'x': 'X', 'y': 'Y'})

            # Convert idx to int to avoid merge issues
            self.idx_points_df['idx'] = pd.to_numeric(self.idx_points_df['idx'], errors='coerce').astype('Int64')

            # Load all lamp height CSV files
            self.lamp_data = {}
            self.available_heights = []

            for file in os.listdir(directory):
                if file.endswith('.csv') and 'led_height_ppfd_values' in file.lower():
                    # Extract height from filename (e.g., "13_cm_led_height_ppfd_values.csv" -> 13)
                    try:
                        height_str = file.split('_')[0]
                        height = int(height_str)

                        # Load the CSV
                        filepath = os.path.join(directory, file)
                        df = pd.read_csv(filepath)

                        # Normalize column names
                        df.columns = df.columns.str.strip().str.lower()

                        # Validate structure (should have idx and LED columns)
                        if 'idx' not in df.columns:
                            messagebox.showwarning("Warning", f"Skipping {file}: missing 'idx' column")
                            continue

                        # Standardize LED column names to: Quantum, 12V, 54V
                        # Map whatever column names exist to standard names
                        column_mapping = {}
                        for col in df.columns:
                            if col == 'idx':
                                continue
                            if 'quantum' in col:
                                column_mapping[col] = 'Quantum'
                            elif '12v' in col or '12v' in col:
                                column_mapping[col] = '12V'
                            elif '54v' in col:
                                column_mapping[col] = '54V'

                        df = df.rename(columns=column_mapping)

                        # Ensure we have the 3 LED types
                        required_leds = {'Quantum', '12V', '54V'}
                        if not required_leds.issubset(set(df.columns)):
                            messagebox.showwarning("Warning",
                                f"Skipping {file}: Expected columns Quantum, 12V, 54V. Found: {list(df.columns)}")
                            continue

                        # Convert idx to int to avoid merge issues
                        df['idx'] = pd.to_numeric(df['idx'], errors='coerce').astype('Int64')

                        # Store data
                        self.lamp_data[height] = df
                        self.available_heights.append(height)

                    except (ValueError, IndexError) as e:
                        messagebox.showwarning("Warning", f"Could not parse height from filename: {file}")
                        continue

            if not self.lamp_data:
                messagebox.showerror("Error", "No valid lamp height CSV files found!")
                return

            # Sort heights
            self.available_heights.sort()

            # === VALIDATION: Check consistency across all files ===
            reference_idx_set = set(self.idx_points_df['idx'].dropna())
            inconsistencies = []

            for height in self.available_heights:
                lamp_df = self.lamp_data[height]
                lamp_idx_set = set(lamp_df['idx'].dropna())

                # Check if idx sets match
                missing_in_lamp = reference_idx_set - lamp_idx_set
                extra_in_lamp = lamp_idx_set - reference_idx_set

                if missing_in_lamp or extra_in_lamp:
                    error_msg = f"Height {height}cm:"
                    if missing_in_lamp:
                        error_msg += f"\n  Missing IDX: {sorted(missing_in_lamp)}"
                    if extra_in_lamp:
                        error_msg += f"\n  Extra IDX: {sorted(extra_in_lamp)}"
                    inconsistencies.append(error_msg)

                # Check for number of measurement points
                if len(lamp_df) != len(self.idx_points_df):
                    inconsistencies.append(
                        f"Height {height}cm: Has {len(lamp_df)} points, expected {len(self.idx_points_df)}"
                    )

            # If inconsistencies found, show error and abort
            if inconsistencies:
                error_message = "DATA INCONSISTENCY DETECTED!\n\n"
                error_message += "All height CSV files must have the same IDX values as idx_to_points.csv\n\n"
                error_message += "Issues found:\n" + "\n".join(inconsistencies)
                error_message += f"\n\nReference (idx_to_points.csv) has {len(self.idx_points_df)} points with IDX: {sorted(reference_idx_set)}"
                messagebox.showerror("Data Validation Error", error_message)

                # Clear loaded data
                self.lamp_data = {}
                self.available_heights = []
                self.idx_points_df = None

                self.data_status_label.config(
                    text="Data validation failed - fix CSV files",
                    foreground='red'
                )
                return

            # Update UI
            self.data_status_label.config(
                text=f"Loaded: {len(self.idx_points_df)} measurement points, {len(self.available_heights)} lamp heights",
                foreground='green'
            )

            # Update estimation tabs status
            self.update_estimation_data_status()
            self.update_ann_estimation_data_status()

            messagebox.showinfo("Success",
                              f"Successfully loaded and validated:\n"
                              f"- {len(self.idx_points_df)} measurement points\n"
                              f"- IDX range: {sorted(reference_idx_set)}\n"
                              f"- {len(self.available_heights)} lamp heights: {self.available_heights}\n\n"
                              f"All files are consistent!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")

    def visualize_all_heights(self):
        """Visualize data for all lamp heights for selected LED type"""
        if not self.lamp_data or self.idx_points_df is None:
            messagebox.showerror("Error", "Please load data first!")
            return

        try:
            # Get selected LED type
            selected_led = self.led_type_var.get()

            # Clear previous content
            for widget in self.content_frame.winfo_children():
                widget.destroy()

            # Create visualization for all heights
            self._create_multi_height_visualization(selected_led)

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Visualization error:\n{str(e)}\n\n{traceback.format_exc()}")

    def _create_multi_height_visualization(self, led_type):
        """Create 3 graphs (one for each height) for selected LED type"""

        # Create main container with two sections: graphs on top, table on bottom
        graph_frame = ttk.Frame(self.content_frame)
        graph_frame.pack(side='top', fill='both', expand=True, pady=5)

        table_frame = ttk.Frame(self.content_frame)
        table_frame.pack(side='bottom', fill='x', pady=10)

        # Create figure with 3 subplots (one for each height)
        fig = Figure(figsize=(18, 6), dpi=100)

        for i, height in enumerate(self.available_heights):
            ax = fig.add_subplot(1, 3, i + 1)

            # Get data for this height
            lamp_df = self.lamp_data[height]

            # Merge with idx_points to get X, Y coordinates
            merged_df = pd.merge(lamp_df, self.idx_points_df, on='idx', how='inner')

            # Get data
            x_coords = merged_df['X'].values
            y_coords = merged_df['Y'].values
            # Multiply by 0.8 to get real PPFD values
            ppfd_values = merged_df[led_type].values * 0.8

            # Calculate real lamp height using dynamic offset
            height_offset = self.get_height_offset(led_type, height)
            real_height = height + height_offset

            # Create grid plot with parallel lines
            self._plot_grid_with_values(ax, x_coords, y_coords, ppfd_values,
                                       led_type, real_height, height)

        fig.suptitle(f'{led_type} LED - PPFD Distribution Across All Heights',
                    fontsize=14, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Embed figure in tkinter
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Create combined data table for all heights
        self._create_combined_table(table_frame, led_type)

    def _plot_grid_with_values(self, ax, x_coords, y_coords, ppfd_values,
                               led_type, real_height, nominal_height):
        """Plot grid with parallel lines, dots, and PPFD values"""

        # Get unique X and Y values to draw grid lines
        unique_x = sorted(set(x_coords))
        unique_y = sorted(set(y_coords))

        # Draw horizontal lines (parallel to X-axis)
        for y in unique_y:
            ax.axhline(y=y, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Draw vertical lines (parallel to Y-axis)
        for x in unique_x:
            ax.axvline(x=x, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Plot measurement points as dots
        scatter = ax.scatter(x_coords, y_coords, c=ppfd_values, cmap='YlOrRd',
                           s=150, edgecolor='black', linewidth=1.2, zorder=5)

        # Add PPFD value labels at each point - smaller font for readability
        # Normalize ppfd values to determine which ones need white text
        ppfd_min, ppfd_max = ppfd_values.min(), ppfd_values.max()
        ppfd_range = ppfd_max - ppfd_min if ppfd_max != ppfd_min else 1

        for x, y, ppfd in zip(x_coords, y_coords, ppfd_values):
            # Use white text for darker backgrounds (higher values), black for lighter
            normalized = (ppfd - ppfd_min) / ppfd_range
            text_color = 'white' if normalized > 0.5 else 'black'

            # Use smaller font (7) and only 0 decimal places for cleaner look
            ax.text(x, y, f'{ppfd:.0f}', ha='center', va='center',
                   fontsize=7, fontweight='bold', color=text_color, zorder=6)

        # Formatting
        ax.set_xlabel('X (cm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (cm)', fontsize=10, fontweight='bold')

        led_specs = self.led_specs[led_type]
        # For Quantum, only show real height to avoid confusion
        if led_type == 'Quantum':
            title = f'{led_type} LED - {real_height}cm\n'
        else:
            title = f'{led_type} LED - {nominal_height}cm\n'
        title += f'LEDs: {led_specs["num_leds"]} | Length: {led_specs["length"]}cm | Cost: {led_specs["cost"]} TL'
        ax.set_title(title, fontsize=10, fontweight='bold')

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='PPFD (µmol/m²/s)', shrink=0.8)

    def _create_combined_table(self, parent, led_type):
        """Create table showing experiment data for all heights"""

        # Title with info
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill='x', pady=5)

        title = ttk.Label(info_frame,
                         text=f"Experiment Data - {led_type} LED - All Heights",
                         font=('Arial', 12, 'bold'))
        title.pack(side='left', padx=10)

        cost = self.led_specs[led_type]['cost']
        height_info = f"Heights: " + ", ".join([f"{h}cm (Real: {h+self.get_height_offset(led_type, h)}cm)" for h in self.available_heights])
        height_info += f" | Cost: {cost} TL"
        ttk.Label(info_frame, text=height_info,
                 font=('Arial', 10, 'italic'), foreground='blue').pack(side='left', padx=20)

        # Create frame with scrollbar
        table_container = ttk.Frame(parent)
        table_container.pack(fill='both', expand=True, padx=10, pady=5)

        # Scrollbars
        scroll_y = ttk.Scrollbar(table_container, orient='vertical')
        scroll_x = ttk.Scrollbar(table_container, orient='horizontal')

        # Create treeview with columns for each height
        columns = ['idx', 'X (cm)', 'Y (cm)']
        for h in self.available_heights:
            columns.append(f'{h}cm PPFD')

        tree = ttk.Treeview(table_container, columns=columns, show='headings',
                           yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set,
                           height=10)

        scroll_y.config(command=tree.yview)
        scroll_x.config(command=tree.xview)

        # Configure columns
        tree.column('idx', width=60, anchor='center')
        tree.column('X (cm)', width=80, anchor='center')
        tree.column('Y (cm)', width=80, anchor='center')
        for h in self.available_heights:
            tree.column(f'{h}cm PPFD', width=120, anchor='center')

        # Set headings
        for col in columns:
            tree.heading(col, text=col)

        # Merge all data
        all_data = {}
        for height in self.available_heights:
            lamp_df = self.lamp_data[height]
            merged_df = pd.merge(lamp_df, self.idx_points_df, on='idx', how='inner')
            for _, row in merged_df.iterrows():
                idx = int(row['idx'])
                if idx not in all_data:
                    all_data[idx] = {'X': row['X'], 'Y': row['Y'], 'ppfd': {}}
                # Multiply by 0.8 to get real PPFD values
                all_data[idx]['ppfd'][height] = row[led_type] * 0.8

        # Insert data
        for idx in sorted(all_data.keys()):
            data = all_data[idx]
            values = [idx, f"{data['X']:.2f}", f"{data['Y']:.2f}"]
            for h in self.available_heights:
                ppfd_val = data['ppfd'].get(h, 0)
                values.append(f"{ppfd_val:.2f}")
            tree.insert('', 'end', values=values)

        # Pack everything
        scroll_y.pack(side='right', fill='y')
        scroll_x.pack(side='bottom', fill='x')
        tree.pack(side='left', fill='both', expand=True)

        # Statistics summary
        stats_frame = ttk.Frame(parent)
        stats_frame.pack(fill='x', padx=10, pady=5)

        stats_text = f"Total Measurement Points: {len(all_data)} | "
        for height in self.available_heights:
            lamp_df = self.lamp_data[height]
            merged_df = pd.merge(lamp_df, self.idx_points_df, on='idx', how='inner')
            # Multiply by 0.8 to get real PPFD values
            ppfd_values = merged_df[led_type].values * 0.8
            # Show real height for Quantum, nominal for others
            display_height = height + self.get_height_offset(led_type, height) if led_type == 'Quantum' else height
            stats_text += f"{display_height}cm: Min={ppfd_values.min():.1f}, "
            stats_text += f"Max={ppfd_values.max():.1f}, "
            stats_text += f"Avg={ppfd_values.mean():.1f} | "

        ttk.Label(stats_frame, text=stats_text[:-3],
                 font=('Arial', 9), foreground='darkgreen').pack()

    def clear_visualization(self):
        """Clear all visualizations"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    # =====================================================
    # PARAMETER ESTIMATION TAB
    # =====================================================

    def create_estimation_interface(self):
        """Create parameter estimation tab interface"""

        # Left panel: Input parameters
        left_panel = ttk.Frame(self.estimation_frame)
        left_panel.pack(side='left', fill='both', padx=10, pady=10, expand=False)

        # Title
        title_label = ttk.Label(left_panel, text="Parameter Estimation",
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Lamp Type Selection
        lamp_type_frame = ttk.LabelFrame(left_panel, text="LED Lamp Type", padding=10)
        lamp_type_frame.pack(fill='x', pady=5)

        ttk.Label(lamp_type_frame, text="Select Lamp Type:").grid(row=0, column=0, sticky='w', pady=5)
        self.est_lamp_type = tk.StringVar(value='Quantum')
        lamp_type_combo = ttk.Combobox(lamp_type_frame, textvariable=self.est_lamp_type,
                                       values=['Quantum', '12V', '54V'], state='readonly', width=13)
        lamp_type_combo.grid(row=0, column=1, padx=5)
        lamp_type_combo.bind('<<ComboboxSelected>>', lambda e: self.update_lamp_specs_display())

        # Display lamp specifications
        self.lamp_specs_label = ttk.Label(lamp_type_frame, text="", font=('Arial', 9, 'italic'), foreground='blue')
        self.lamp_specs_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=5)
        self.update_lamp_specs_display()

        # Lamp Orientation
        orientation_frame = ttk.LabelFrame(left_panel, text="Lamp Orientation", padding=10)
        orientation_frame.pack(fill='x', pady=5)

        self.est_orientation = tk.StringVar(value='perpendicular')
        ttk.Radiobutton(orientation_frame, text="Perpendicular to Channels",
                       variable=self.est_orientation, value='perpendicular',
                       command=self.update_est_led_axis_indicator).pack(anchor='w', pady=3)
        ttk.Radiobutton(orientation_frame, text="Parallel to Channels",
                       variable=self.est_orientation, value='parallel',
                       command=self.update_est_led_axis_indicator).pack(anchor='w', pady=3)

        # LED Axis Indicator
        self.est_led_axis_label = ttk.Label(orientation_frame, text="LEDs along X-axis →",
                                           font=('Arial', 9, 'italic'), foreground='blue')
        self.est_led_axis_label.pack(anchor='w', pady=8)

        # Data Status Info
        data_status_frame = ttk.LabelFrame(left_panel, text="Data Information", padding=10)
        data_status_frame.pack(fill='x', pady=5)

        self.est_data_status_label = ttk.Label(data_status_frame,
                                               text="Load visualization data first",
                                               font=('Arial', 9, 'italic'),
                                               foreground='orange',
                                               wraplength=200)
        self.est_data_status_label.pack(anchor='w', pady=5)

        # Height Selection for Training and Testing
        height_selection_frame = ttk.LabelFrame(left_panel, text="Height Selection", padding=10)
        height_selection_frame.pack(fill='x', pady=5)

        # Training Heights Selection
        ttk.Label(height_selection_frame, text="Training Heights:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky='w', pady=5)

        ttk.Label(height_selection_frame, text="Height 1:").grid(row=1, column=0, sticky='w', padx=10)
        self.train_height1 = tk.StringVar()
        self.train_height1_combo = ttk.Combobox(height_selection_frame, textvariable=self.train_height1,
                                                state='readonly', width=15)
        self.train_height1_combo.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(height_selection_frame, text="Height 2:").grid(row=2, column=0, sticky='w', padx=10)
        self.train_height2 = tk.StringVar()
        self.train_height2_combo = ttk.Combobox(height_selection_frame, textvariable=self.train_height2,
                                                state='readonly', width=15)
        self.train_height2_combo.grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(height_selection_frame, text="Height 3:").grid(row=3, column=0, sticky='w', padx=10)
        self.train_height3 = tk.StringVar()
        self.train_height3_combo = ttk.Combobox(height_selection_frame, textvariable=self.train_height3,
                                                state='readonly', width=15)
        self.train_height3_combo.grid(row=3, column=1, padx=5, pady=2)

        # Testing Height Selection
        ttk.Label(height_selection_frame, text="Testing Height:", font=('Arial', 9, 'bold')).grid(row=4, column=0, sticky='w', pady=(10, 5))

        self.test_height = tk.StringVar()
        self.test_height_combo = ttk.Combobox(height_selection_frame, textvariable=self.test_height,
                                             state='readonly', width=15)
        self.test_height_combo.grid(row=4, column=1, padx=5, pady=2)

        # Note about real heights
        self.height_note_label = ttk.Label(height_selection_frame, text="",
                                           font=('Arial', 8, 'italic'), foreground='gray',
                                           wraplength=280)
        self.height_note_label.grid(row=5, column=0, columnspan=2, sticky='w', pady=5)

        # Search method section
        search_frame = ttk.LabelFrame(left_panel, text="Search Method", padding=10)
        search_frame.pack(fill='x', pady=5)

        self.search_method = tk.StringVar(value='golden')
        ttk.Radiobutton(search_frame, text="Golden Section",
                       variable=self.search_method, value='golden').pack(anchor='w', pady=3)
        ttk.Radiobutton(search_frame, text="Grid Search",
                       variable=self.search_method, value='grid').pack(anchor='w', pady=3)

        # Grid search parameters
        grid_param_frame = ttk.LabelFrame(search_frame, text="Grid Search Parameters", padding=5)
        grid_param_frame.pack(fill='x', pady=5)

        ttk.Label(grid_param_frame, text="Min α (cm⁻¹):").grid(row=0, column=0, sticky='w', pady=3)
        self.alpha_min = tk.DoubleVar(value=0.0)
        ttk.Entry(grid_param_frame, textvariable=self.alpha_min, width=12).grid(row=0, column=1)

        ttk.Label(grid_param_frame, text="Max α (cm⁻¹):").grid(row=1, column=0, sticky='w', pady=3)
        self.alpha_max = tk.DoubleVar(value=0.1)
        ttk.Entry(grid_param_frame, textvariable=self.alpha_max, width=12).grid(row=1, column=1)

        ttk.Label(grid_param_frame, text="Number of Points:").grid(row=2, column=0, sticky='w', pady=3)
        self.n_alpha = tk.IntVar(value=1000)
        ttk.Entry(grid_param_frame, textvariable=self.n_alpha, width=12).grid(row=2, column=1)

        # Run Estimation button
        ttk.Button(left_panel, text="Run Estimation", command=self.run_estimation,
                  style='Accent.TButton').pack(fill='x', pady=15)

        # Right panel: Results
        self.right_panel_est = ttk.Frame(self.estimation_frame)
        self.right_panel_est.pack(side='right', fill='both', expand=True, padx=10, pady=10)

    def update_lamp_specs_display(self):
        """Update lamp specifications display"""
        lamp_type = self.est_lamp_type.get()
        specs = self.led_specs[lamp_type]
        text = f"LEDs: {specs['num_leds']} | Length: {specs['length']}cm | Cost: {specs['cost']} TL"
        if lamp_type == 'Quantum':
            text += f" | Height offset: 13cm→0cm, 27/35cm→+4cm"
        self.lamp_specs_label.config(text=text)

        # Update data status to show real heights for the selected lamp type
        self.update_estimation_data_status()

    def update_est_led_axis_indicator(self):
        """Update LED axis indicator in estimation tab"""
        orientation = self.est_orientation.get()
        if orientation == 'perpendicular':
            self.est_led_axis_label.config(text="LEDs along X-axis →", foreground='blue')
        else:
            self.est_led_axis_label.config(text="LEDs along Y-axis ↑", foreground='green')

    def update_estimation_data_status(self):
        """Update data status in estimation tab and populate height selection comboboxes"""
        # Check if estimation tab UI has been created
        if not hasattr(self, 'est_data_status_label'):
            return

        if self.lamp_data and self.idx_points_df is not None:
            text = f"Data loaded: {len(self.idx_points_df)} points, {len(self.available_heights)} heights\n"
            text += f"Nominal Heights: {self.available_heights}\n"

            # Show real heights for each lamp type
            lamp_type = self.est_lamp_type.get()
            real_heights = [h + self.get_height_offset(lamp_type, h) for h in self.available_heights]

            if lamp_type == 'Quantum':
                text += f"Real Heights ({lamp_type}): {real_heights} (13cm→0, 27/35cm→+4cm)"
            else:
                text += f"Real Heights ({lamp_type}): {real_heights}"

            self.est_data_status_label.config(text=text, foreground='green')

            # Populate height selection comboboxes with real heights
            height_options = [f"{h}cm (Real: {h + self.get_height_offset(lamp_type, h)}cm)" if lamp_type == 'Quantum' else f"{h}cm"
                            for h in self.available_heights]

            # All training and testing heights can be None
            self.train_height1_combo['values'] = ['None'] + height_options
            self.train_height2_combo['values'] = ['None'] + height_options
            self.train_height3_combo['values'] = ['None'] + height_options
            self.test_height_combo['values'] = ['None'] + height_options

            # Set default selections if not already set
            if not self.train_height1.get() and len(height_options) >= 1:
                self.train_height1.set(height_options[0])
            if not self.train_height2.get():
                if len(height_options) >= 2:
                    self.train_height2.set(height_options[1])
                else:
                    self.train_height2.set('None')
            if not self.train_height3.get():
                if len(height_options) >= 3:
                    self.train_height3.set(height_options[2])
                else:
                    self.train_height3.set('None')
            if not self.test_height.get():
                self.test_height.set('None')

            # Update note about height offset
            if lamp_type == 'Quantum':
                self.height_note_label.config(
                    text=f"Note: For {lamp_type}, 13cm→real=13cm, 27cm→real=31cm, 35cm→real=39cm",
                    foreground='blue')
            else:
                self.height_note_label.config(text="")

        else:
            self.est_data_status_label.config(text="No data loaded. Please load data from Visualization tab first.",
                                            foreground='orange')
            # Clear comboboxes
            if hasattr(self, 'train_height1_combo'):
                self.train_height1_combo['values'] = []
                self.train_height2_combo['values'] = []
                self.train_height3_combo['values'] = []
                self.test_height_combo['values'] = []

    def run_estimation(self):
        """Run parameter estimation using CSV data"""
        try:
            # Check if data is loaded
            if not self.lamp_data or self.idx_points_df is None:
                messagebox.showerror("Error", "Please load data from the Visualization tab first!")
                return

            # Get lamp configuration
            lamp_type = self.est_lamp_type.get()
            lamp_specs = self.led_specs[lamp_type]
            lamp_length = lamp_specs['length']
            led_number = lamp_specs['num_leds']
            orientation = self.est_orientation.get()

            # Parse selected heights (extract nominal height from "XXcm" or "XXcm (Real: YYcm)")
            def parse_height(height_str):
                if height_str == 'None' or not height_str:
                    return None
                return int(height_str.split('cm')[0])

            train_height_1 = parse_height(self.train_height1.get())
            train_height_2 = parse_height(self.train_height2.get())
            train_height_3 = parse_height(self.train_height3.get())
            test_height = parse_height(self.test_height.get())

            # Build training heights list (exclude None)
            train_heights = []
            if train_height_1 is not None:
                train_heights.append(train_height_1)
            if train_height_2 is not None:
                train_heights.append(train_height_2)
            if train_height_3 is not None:
                train_heights.append(train_height_3)

            # Check that at least one training height is selected
            if len(train_heights) == 0:
                messagebox.showerror("Error", "Please select at least one Training Height!")
                return

            # Calculate real heights using dynamic offset
            train_real_heights = [h + self.get_height_offset(lamp_type, h) for h in train_heights]

            info_msg = f"Lamp Type: {lamp_type}\n"
            info_msg += f"Orientation: {orientation}\n\n"
            info_msg += f"Training Heights:\n"
            info_msg += f"  Nominal: {train_heights} cm\n"
            info_msg += f"  Real: {train_real_heights} cm\n\n"

            if test_height is not None:
                test_real_height = test_height + self.get_height_offset(lamp_type, test_height)
                info_msg += f"Testing Height:\n"
                info_msg += f"  Nominal: {test_height} cm\n"
                info_msg += f"  Real: {test_real_height} cm"
            else:
                info_msg += f"Testing: None (No testing will be performed)"

            if lamp_type == 'Quantum':
                info_msg += f"\n\n(Height offset for {lamp_type}: 13cm→0, 27/35cm→+4cm)"

            messagebox.showinfo("Estimation Setup", info_msg)

            # Prepare training data from min and max heights
            train_x_coords = []
            train_y_coords = []
            train_lamp_heights = []
            train_ppfd = []
            train_idx = []

            for height in train_heights:
                lamp_df = self.lamp_data[height]
                merged_df = pd.merge(lamp_df, self.idx_points_df, on='idx', how='inner')

                x_coords = merged_df['X'].values
                y_coords = merged_df['Y'].values
                ppfd_values = merged_df[lamp_type].values * 0.8  # Real PPFD values
                idx_values = merged_df['idx'].values

                # Real lamp height (add dynamic offset)
                real_height = height + self.get_height_offset(lamp_type, height)

                for x, y, ppfd, idx in zip(x_coords, y_coords, ppfd_values, idx_values):
                    train_x_coords.append(x)
                    train_y_coords.append(y)
                    train_lamp_heights.append(real_height)
                    train_ppfd.append(ppfd)
                    train_idx.append(idx)

            train_x_coords = np.array(train_x_coords)
            train_y_coords = np.array(train_y_coords)
            train_lamp_heights = np.array(train_lamp_heights)
            train_ppfd = np.array(train_ppfd)
            train_idx = np.array(train_idx)
            n_train = len(train_x_coords)

            # Prepare testing data (only if test_height is not None)
            if test_height is not None:
                test_lamp_df = self.lamp_data[test_height]
                test_merged_df = pd.merge(test_lamp_df, self.idx_points_df, on='idx', how='inner')

                test_x_coords = test_merged_df['X'].values
                test_y_coords = test_merged_df['Y'].values
                test_ppfd = test_merged_df[lamp_type].values * 0.8
                test_idx = test_merged_df['idx'].values
                test_real_height = test_height + self.get_height_offset(lamp_type, test_height)
                test_lamp_heights = np.full(len(test_x_coords), test_real_height)
                n_test = len(test_x_coords)
            else:
                # No testing data
                test_x_coords = None
                test_y_coords = None
                test_ppfd = None
                test_idx = None
                test_lamp_heights = None
                n_test = 0

            # Create LED positions
            if orientation == 'perpendicular':
                x_leds = np.linspace(-lamp_length/2, lamp_length/2, led_number)
                y_leds = np.zeros(led_number)
            else:
                x_leds = np.zeros(led_number)
                y_leds = np.linspace(-lamp_length/2, lamp_length/2, led_number)

            # Calculate distance matrix for TRAINING data
            R_matrix_train = np.zeros((led_number, n_train))
            for j in range(n_train):
                z_leds_j = train_lamp_heights[j]
                for i in range(led_number):
                    dx = train_x_coords[j] - x_leds[i]
                    dy = train_y_coords[j] - y_leds[i]
                    dz = 0 - z_leds_j  # Measurement at z=0, LEDs at z=lamp_height
                    R_matrix_train[i, j] = np.sqrt(dx**2 + dy**2 + dz**2)

            # Calculate distance matrix for TESTING data (only if test data exists)
            if test_height is not None:
                R_matrix_test = np.zeros((led_number, n_test))
                for j in range(n_test):
                    z_leds_j = test_lamp_heights[j]
                    for i in range(led_number):
                        dx = test_x_coords[j] - x_leds[i]
                        dy = test_y_coords[j] - y_leds[i]
                        dz = 0 - z_leds_j
                        R_matrix_test[i, j] = np.sqrt(dx**2 + dy**2 + dz**2)
            else:
                R_matrix_test = None

            # Define cost functions using TRAINING data
            def calculate_optimal_theta1(alpha):
                S_train = np.sum(np.exp(-alpha * R_matrix_train), axis=0)
                numerator = np.dot(S_train, train_ppfd)
                denominator = np.dot(S_train, S_train)
                theta1_opt = numerator / denominator if denominator > 0 else 0
                return theta1_opt, S_train

            def calculate_sse(alpha):
                theta1_opt, S_train = calculate_optimal_theta1(alpha)
                ppfd_train_pred = theta1_opt * S_train
                residuals = train_ppfd - ppfd_train_pred
                sse = np.sum(residuals**2)
                return sse, theta1_opt, ppfd_train_pred

            # Run estimation
            search_method = self.search_method.get()

            if search_method == 'grid':
                alpha_min = self.alpha_min.get()
                alpha_max = self.alpha_max.get()
                n_alpha = self.n_alpha.get()
                alpha_values = np.linspace(alpha_min, alpha_max, n_alpha)
                sse_values = np.zeros(n_alpha)
                theta1_values = np.zeros(n_alpha)

                for i, alpha in enumerate(alpha_values):
                    sse, theta1, _ = calculate_sse(alpha)
                    sse_values[i] = sse
                    theta1_values[i] = theta1

                min_idx = np.argmin(sse_values)
                alpha_result = alpha_values[min_idx]
                theta1_result = theta1_values[min_idx]
                sse_result = sse_values[min_idx]

                # Store grid search data for plotting
                self.grid_search_data = {
                    'alpha_values': alpha_values,
                    'sse_values': sse_values,
                    'theta1_values': theta1_values
                }

            else:  # Golden section
                alpha_max = self.alpha_max.get()

                def cost_func(alpha):
                    sse, _, _ = calculate_sse(alpha)
                    return sse

                result = minimize_scalar(cost_func, bounds=(0, alpha_max), method='bounded',
                                        options={'xatol': 1e-12})

                alpha_result = result.x
                sse_result = result.fun
                theta1_result, _ = calculate_optimal_theta1(alpha_result)

            # Calculate predictions for TRAINING data
            theta1_check, S_train = calculate_optimal_theta1(alpha_result)
            train_ppfd_predicted = theta1_check * S_train
            train_residuals = train_ppfd - train_ppfd_predicted

            # Calculate training R² and RMSE
            ss_res_train = np.sum(train_residuals**2)
            ss_tot_train = np.sum((train_ppfd - train_ppfd.mean())**2)
            r_squared_train = 1 - (ss_res_train / ss_tot_train) if ss_tot_train > 0 else 0
            rmse_train = np.sqrt(ss_res_train / n_train)

            # Calculate predictions for TESTING data (only if test data exists)
            if test_height is not None:
                S_test = np.sum(np.exp(-alpha_result * R_matrix_test), axis=0)
                test_ppfd_predicted = theta1_result * S_test
                test_residuals = test_ppfd - test_ppfd_predicted

                # Calculate testing R² and RMSE
                ss_res_test = np.sum(test_residuals**2)
                ss_tot_test = np.sum((test_ppfd - test_ppfd.mean())**2)
                r_squared_test = 1 - (ss_res_test / ss_tot_test) if ss_tot_test > 0 else 0
                rmse_test = np.sqrt(ss_res_test / n_test)
            else:
                # No testing data
                test_ppfd_predicted = None
                test_residuals = None
                r_squared_test = None
                rmse_test = None

            # Store results
            self.est_results = {
                'alpha': alpha_result,
                'theta1': theta1_result,
                'sse': sse_result,
                'search_method': search_method,
                'lamp_type': lamp_type,
                'orientation': orientation,
                'train_heights': train_heights,
                'test_height': test_height,
                # Training data
                'train_x_coords': train_x_coords,
                'train_y_coords': train_y_coords,
                'train_lamp_heights': train_lamp_heights,
                'train_ppfd_measured': train_ppfd,
                'train_ppfd_predicted': train_ppfd_predicted,
                'train_residuals': train_residuals,
                'train_idx': train_idx,
                'train_r_squared': r_squared_train,
                'train_rmse': rmse_train,
                'n_train': n_train,
                # Testing data
                'test_x_coords': test_x_coords,
                'test_y_coords': test_y_coords,
                'test_lamp_heights': test_lamp_heights,
                'test_ppfd_measured': test_ppfd,
                'test_ppfd_predicted': test_ppfd_predicted,
                'test_residuals': test_residuals,
                'test_idx': test_idx,
                'test_r_squared': r_squared_test,
                'test_rmse': rmse_test,
                'n_test': n_test
            }

            # Visualize results
            self.visualize_estimation_results()

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Estimation error:\n{str(e)}\n\n{traceback.format_exc()}")

    def visualize_estimation_results(self):
        """Visualize estimation results using the new visualization style"""
        results = self.est_results

        # Clear previous results
        for widget in self.right_panel_est.winfo_children():
            widget.destroy()

        # Create canvas with scrollbar
        canvas = tk.Canvas(self.right_panel_est, bg='white')
        scrollbar = ttk.Scrollbar(self.right_panel_est, orient="vertical", command=canvas.yview)

        # Create a frame inside the canvas
        scrollable_frame = ttk.Frame(canvas)

        # Configure the canvas
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Main container
        main_container = scrollable_frame

        # Top section: Summary and stats
        summary_frame = ttk.LabelFrame(main_container, text="Estimation Summary", padding=10)
        summary_frame.pack(fill='x', padx=10, pady=10)

        # Create two columns for summary
        left_summary = ttk.Frame(summary_frame)
        left_summary.pack(side='left', fill='both', expand=True)

        right_summary = ttk.Frame(summary_frame)
        right_summary.pack(side='right', fill='both', expand=True)

        # Get lamp specs for display
        lamp_type = results['lamp_type']
        train_real_heights = [h + self.get_height_offset(lamp_type, h) for h in results['train_heights']]
        test_height = results['test_height']

        # Left column: Parameters
        ttk.Label(left_summary, text=f"Lamp Type: {lamp_type}", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(left_summary, text=f"Orientation: {results['orientation'].upper()}", font=('Arial', 10)).pack(anchor='w', pady=2)

        # Show heights
        ttk.Label(left_summary, text=f"Training Heights (nominal): {results['train_heights']} cm", font=('Arial', 9)).pack(anchor='w', pady=1)
        if lamp_type == 'Quantum':
            ttk.Label(left_summary, text=f"Training Heights (real): {train_real_heights} cm", font=('Arial', 9, 'bold'), foreground='blue').pack(anchor='w', pady=1)

        if test_height is not None:
            ttk.Label(left_summary, text=f"Testing Height (nominal): {test_height} cm", font=('Arial', 9)).pack(anchor='w', pady=1)
            if lamp_type == 'Quantum':
                test_real_height = test_height + self.get_height_offset(lamp_type, test_height)
                ttk.Label(left_summary, text=f"Testing Height (real): {test_real_height} cm", font=('Arial', 9, 'bold'), foreground='blue').pack(anchor='w', pady=1)
        else:
            ttk.Label(left_summary, text=f"Testing Height: None", font=('Arial', 9), foreground='gray').pack(anchor='w', pady=1)

        ttk.Label(left_summary, text="", font=('Arial', 2)).pack(anchor='w', pady=1)  # Spacer
        ttk.Label(left_summary, text=f"θ₁ (PPFD per LED): {results['theta1']:.8f} µmol/(m²·s)", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(left_summary, text=f"α (Decay Rate): {results['alpha']:.8f} cm⁻¹", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)

        # Right column: Quality metrics
        ttk.Label(right_summary, text=f"Search Method: {results['search_method'].upper()}", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(right_summary, text="Training Results:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(right_summary, text=f"  R²: {results['train_r_squared']:.6f} | RMSE: {results['train_rmse']:.4f}", font=('Arial', 9)).pack(anchor='w', padx=10, pady=1)
        ttk.Label(right_summary, text=f"  Points: {results['n_train']}", font=('Arial', 9)).pack(anchor='w', padx=10, pady=1)

        if test_height is not None:
            ttk.Label(right_summary, text="Testing Results:", font=('Arial', 10, 'bold'), foreground='darkgreen').pack(anchor='w', pady=2)
            ttk.Label(right_summary, text=f"  R²: {results['test_r_squared']:.6f} | RMSE: {results['test_rmse']:.4f}", font=('Arial', 9), foreground='darkgreen').pack(anchor='w', padx=10, pady=1)
            ttk.Label(right_summary, text=f"  Points: {results['n_test']}", font=('Arial', 9), foreground='darkgreen').pack(anchor='w', padx=10, pady=1)
        else:
            ttk.Label(right_summary, text="Testing Results: None", font=('Arial', 10, 'bold'), foreground='gray').pack(anchor='w', pady=2)

        # Section: Equation
        eq_frame = ttk.LabelFrame(main_container, text="Estimated Model", padding=10)
        eq_frame.pack(fill='x', padx=10, pady=10)

        equation_text = self._create_equation_text(results)
        eq_label = ttk.Label(eq_frame, text=equation_text, font=('Courier', 10),
                            background='lightblue', relief='solid', padding=10)
        eq_label.pack(fill='x')

        # Add interactive estimation section
        self._create_interactive_estimation(eq_frame, results)

        # Middle section: Graphs (maksimum 2 yan yana)
        graph_frame = ttk.Frame(main_container)
        graph_frame.pack(fill='x', padx=10, pady=10)

        # Get the individual heights
        train_heights = results['train_heights']
        test_height = results['test_height']
        lamp_type = results['lamp_type']

        # Split training data by height
        train_data_by_height = self._split_training_data_by_height(results)

        # Prepare all data (training + testing if exists)
        all_plot_data = []
        for i in range(len(train_heights)):
            all_plot_data.append({
                'data': train_data_by_height[i],
                'nominal_height': train_heights[i],
                'title': f'Training {i+1}'
            })

        # Add testing data (only if test_height is not None)
        if test_height is not None:
            test_data = {
                'x_coords': results['test_x_coords'],
                'y_coords': results['test_y_coords'],
                'measured': results['test_ppfd_measured'],
                'predicted': results['test_ppfd_predicted'],
                'idx': results['test_idx']
            }
            all_plot_data.append({
                'data': test_data,
                'nominal_height': test_height,
                'title': 'Testing'
            })

        # Create figures with max 2 plots per row
        num_total_plots = len(all_plot_data)
        plots_per_row = 2
        num_rows = (num_total_plots + plots_per_row - 1) // plots_per_row  # Ceiling division

        for row in range(num_rows):
            # Create a figure for this row
            plots_in_this_row = min(plots_per_row, num_total_plots - row * plots_per_row)
            fig = Figure(figsize=(16, 6), dpi=90)

            for col in range(plots_in_this_row):
                plot_idx = row * plots_per_row + col
                plot_info = all_plot_data[plot_idx]

                ax = fig.add_subplot(1, plots_per_row, col + 1)
                self._plot_height_comparison(ax, plot_info['data'], plot_info['nominal_height'],
                                            lamp_type, plot_info['title'])

            fig.tight_layout()

            # Embed figure
            fig_canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            fig_canvas.draw()
            fig_canvas.get_tk_widget().pack(fill='x', expand=False, pady=5)

        # Additional section: IDX vs Predicted/Measured line plots
        line_plot_frame = ttk.Frame(main_container)
        line_plot_frame.pack(fill='x', padx=10, pady=10)

        self._create_idx_line_plots(line_plot_frame, results, train_data_by_height, train_heights, lamp_type)

        # Bottom section: Detailed Tables for each height
        table_frame = ttk.Frame(main_container)
        table_frame.pack(fill='x', padx=10, pady=10)

        self._create_detailed_tables(table_frame, results)

        # Show success message
        success_msg = f"Parameter estimation completed!\n\n"
        success_msg += f"θ₁ = {results['theta1']:.8f}\n"
        success_msg += f"α = {results['alpha']:.8f}\n\n"
        success_msg += f"Training - R²: {results['train_r_squared']:.6f} | RMSE: {results['train_rmse']:.4f}\n"

        if results['test_height'] is not None:
            success_msg += f"Testing  - R²: {results['test_r_squared']:.6f} | RMSE: {results['test_rmse']:.4f}"
        else:
            success_msg += f"Testing: None (No testing performed)"

        messagebox.showinfo("Success", success_msg)

    def _create_equation_text(self, results):
        """Create the equation text showing the estimated model"""
        theta1 = results['theta1']
        alpha = results['alpha']
        lamp_type = results['lamp_type']
        led_count = self.led_specs[lamp_type]['num_leds']

        # Create equation text
        eq_text = f"Estimated Model Equation:\n\n"
        eq_text += f"PPFD(x,y,h) = Σ θ₁ × exp(-α × R_i)\n\n"
        eq_text += f"where: θ₁ = {theta1:.6f} µmol/(m²·s)  |  α = {alpha:.6f} cm⁻¹  |  LEDs = {led_count}\n"
        eq_text += f"R_i = √[(x - x_i)² + (y - y_i)² + h²]"

        return eq_text

    def _create_interactive_estimation(self, parent, results):
        """Create interactive estimation interface for custom x, y, h values"""
        # Create a frame for the interactive section
        interactive_frame = ttk.Frame(parent)
        interactive_frame.pack(fill='x', pady=10)

        # Title
        ttk.Label(interactive_frame, text="Interactive PPFD Estimation",
                 font=('Arial', 11, 'bold')).grid(row=0, column=0, columnspan=6, pady=5)

        # Input fields
        ttk.Label(interactive_frame, text="X (cm):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        x_entry = ttk.Entry(interactive_frame, width=10)
        x_entry.grid(row=1, column=1, padx=5, pady=5)
        x_entry.insert(0, "0")

        ttk.Label(interactive_frame, text="Y (cm):").grid(row=1, column=2, padx=5, pady=5, sticky='e')
        y_entry = ttk.Entry(interactive_frame, width=10)
        y_entry.grid(row=1, column=3, padx=5, pady=5)
        y_entry.insert(0, "0")

        ttk.Label(interactive_frame, text="H (cm):").grid(row=1, column=4, padx=5, pady=5, sticky='e')
        h_entry = ttk.Entry(interactive_frame, width=10)
        h_entry.grid(row=1, column=5, padx=5, pady=5)
        # Default to the first training height
        lamp_type = results['lamp_type']
        first_train_height = results['train_heights'][0]
        default_h = first_train_height + self.get_height_offset(lamp_type, first_train_height)
        h_entry.insert(0, str(default_h))

        # Result display
        result_label = ttk.Label(interactive_frame, text="Predicted PPFD: --",
                                font=('Arial', 12, 'bold'), foreground='darkgreen',
                                background='lightyellow', relief='solid', padding=10)
        result_label.grid(row=2, column=0, columnspan=6, pady=10, sticky='ew')

        # Calculate button
        def calculate_ppfd():
            try:
                x_val = float(x_entry.get())
                y_val = float(y_entry.get())
                h_val = float(h_entry.get())

                # Get model parameters
                theta1 = results['theta1']
                alpha = results['alpha']
                lamp_type = results['lamp_type']
                orientation = results['orientation']

                # Get lamp specifications
                lamp_specs = self.led_specs[lamp_type]
                lamp_length = lamp_specs['length']
                led_number = lamp_specs['num_leds']

                # Create LED positions based on orientation
                if orientation == 'perpendicular':
                    x_leds = np.linspace(-lamp_length/2, lamp_length/2, led_number)
                    y_leds = np.zeros(led_number)
                else:  # parallel
                    x_leds = np.zeros(led_number)
                    y_leds = np.linspace(-lamp_length/2, lamp_length/2, led_number)

                # Calculate PPFD using the model
                ppfd_sum = 0
                for i in range(led_number):
                    dx = x_val - x_leds[i]
                    dy = y_val - y_leds[i]
                    dz = h_val  # Distance from measurement point (z=0) to LED at height h
                    R = np.sqrt(dx**2 + dy**2 + dz**2)
                    ppfd_sum += theta1 * np.exp(-alpha * R)

                # Update result label
                result_label.config(
                    text=f"Predicted PPFD at ({x_val:.1f}, {y_val:.1f}, h={h_val:.1f}cm): {ppfd_sum:.2f} µmol/m²/s",
                    foreground='darkgreen'
                )

            except ValueError as e:
                result_label.config(
                    text="Error: Please enter valid numeric values!",
                    foreground='red'
                )
            except Exception as e:
                result_label.config(
                    text=f"Error: {str(e)}",
                    foreground='red'
                )

        calc_button = ttk.Button(interactive_frame, text="Calculate PPFD",
                                command=calculate_ppfd)
        calc_button.grid(row=3, column=0, columnspan=6, pady=5)

        # Add note about coordinate system
        note_text = f"Note: Enter coordinates relative to lamp center. "
        note_text += f"Orientation: {results['orientation'].upper()}. "
        note_text += f"H is the lamp height (real height including offset)."
        ttk.Label(interactive_frame, text=note_text,
                 font=('Arial', 8, 'italic'), foreground='gray',
                 wraplength=600).grid(row=4, column=0, columnspan=6, pady=5)

    def _create_idx_line_plots(self, parent, results, train_data_by_height, train_heights, lamp_type):
        """Create line plots showing IDX vs Measured/Predicted for each height"""

        test_height = results['test_height']

        # Prepare all plot data
        all_line_data = []

        # Add training heights
        for i in range(len(train_heights)):
            data = train_data_by_height[i]
            real_h = train_heights[i] + self.get_height_offset(lamp_type, train_heights[i])
            all_line_data.append({
                'idx': data['idx'],
                'measured': data['measured'],
                'predicted': data['predicted'],
                'title': f'Training {i+1} (H={real_h}cm)',
                'color': 'blue'
            })

        # Add testing data if exists
        if test_height is not None:
            test_real_h = test_height + self.get_height_offset(lamp_type, test_height)

            # Support both Linear Regression and ANN result formats
            if 'test_ppfd_measured' in results:
                # Linear Regression format
                test_idx = results['test_idx']
                test_measured = results['test_ppfd_measured']
                test_predicted = results['test_ppfd_predicted']
            else:
                # ANN format
                test_idx = results['test_idx']
                test_measured = results['y_test']
                test_predicted = results['y_test_pred']

            all_line_data.append({
                'idx': test_idx,
                'measured': test_measured,
                'predicted': test_predicted,
                'title': f'Testing (H={test_real_h}cm)',
                'color': 'green'
            })

        # Create plots (max 2 per row)
        num_plots = len(all_line_data)
        plots_per_row = 2
        num_rows = (num_plots + plots_per_row - 1) // plots_per_row

        for row in range(num_rows):
            plots_in_this_row = min(plots_per_row, num_plots - row * plots_per_row)
            fig = Figure(figsize=(16, 5), dpi=90)

            for col in range(plots_in_this_row):
                plot_idx = row * plots_per_row + col
                plot_data = all_line_data[plot_idx]

                ax = fig.add_subplot(1, plots_per_row, col + 1)

                # Sort by IDX for proper line plotting
                sorted_indices = np.argsort(plot_data['idx'])
                idx_sorted = plot_data['idx'][sorted_indices]
                measured_sorted = plot_data['measured'][sorted_indices]
                predicted_sorted = plot_data['predicted'][sorted_indices]

                # Plot lines
                ax.plot(idx_sorted, measured_sorted, 'o-', color=plot_data['color'],
                       linewidth=2, markersize=6, label='Measured', alpha=0.7)
                ax.plot(idx_sorted, predicted_sorted, 's--', color='red',
                       linewidth=2, markersize=6, label='Predicted', alpha=0.7)

                ax.set_xlabel('Measurement IDX', fontsize=11, fontweight='bold')
                ax.set_ylabel('PPFD (µmol/m²/s)', fontsize=11, fontweight='bold')
                ax.set_title(plot_data['title'], fontsize=12, fontweight='bold')
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.tick_params(labelsize=10)

            fig.tight_layout()

            # Embed figure
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='x', expand=False, pady=5)

    def _split_training_data_by_height(self, results):
        """Split training data by individual heights"""
        train_heights = results['train_heights']
        train_x = results['train_x_coords']
        train_y = results['train_y_coords']
        train_lamp_h = results['train_lamp_heights']
        train_measured = results['train_ppfd_measured']
        train_predicted = results['train_ppfd_predicted']
        train_idx = results['train_idx']

        # Get real heights
        lamp_type = results['lamp_type']
        real_heights = [h + self.get_height_offset(lamp_type, h) for h in train_heights]

        # Split by height
        data_by_height = []
        for real_h in real_heights:
            indices = np.where(train_lamp_h == real_h)[0]
            data_by_height.append({
                'x_coords': train_x[indices],
                'y_coords': train_y[indices],
                'measured': train_measured[indices],
                'predicted': train_predicted[indices],
                'idx': train_idx[indices]
            })

        return data_by_height

    def _plot_height_comparison(self, ax, data, nominal_height, lamp_type, data_type):
        """Plot measured vs predicted for a single height using grid style"""
        x_coords = data['x_coords']
        y_coords = data['y_coords']
        measured = data['measured']
        predicted = data['predicted']
        idx_vals = data['idx']

        # Get unique X and Y for grid lines
        unique_x = sorted(set(x_coords))
        unique_y = sorted(set(y_coords))

        # Draw grid lines
        for y in unique_y:
            ax.axhline(y=y, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)
        for x in unique_x:
            ax.axvline(x=x, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Calculate errors
        errors = predicted - measured
        abs_errors = np.abs(errors)

        # Plot points with colors based on error - REVERSE colormap for better readability
        # YlGn_r: yellow (low error) to green (high error), reversed
        scatter = ax.scatter(x_coords, y_coords, c=abs_errors, cmap='YlGn',
                           s=250, edgecolor='black', linewidth=1.5, zorder=5)

        # Add text labels showing IDX, M (measured) and P (predicted)
        # Use WHITE text for better visibility on colored backgrounds
        for i in range(len(x_coords)):
            ax.text(x_coords[i], y_coords[i], f'IDX:{int(idx_vals[i])}\nM:{measured[i]:.0f}\nP:{predicted[i]:.0f}',
                   ha='center', va='center', fontsize=7, fontweight='bold', color='white',
                   zorder=6, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))

        # Title and labels
        height_offset = self.get_height_offset(lamp_type, nominal_height)
        real_height = nominal_height + height_offset
        if lamp_type == 'Quantum':
            title = f'{data_type}: {nominal_height}cm (Real: {real_height}cm)'
        else:
            title = f'{data_type}: {nominal_height}cm'

        ax.set_xlabel('X (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (cm)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

        # Increase tick label size
        ax.tick_params(labelsize=10)

        # Add colorbar with larger font
        cbar = plt.colorbar(scatter, ax=ax, label='Absolute Error (µmol/m²/s)', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

    def _plot_measured_vs_predicted_scatter(self, ax, results, mode='train'):
        """Plot measured vs predicted as scatter with identity line"""
        if mode == 'train':
            ppfd_measured = results['train_ppfd_measured']
            ppfd_predicted = results['train_ppfd_predicted']
            r_squared = results['train_r_squared']
            title_prefix = "TRAINING"
            color = 'blue'
        else:
            ppfd_measured = results['test_ppfd_measured']
            ppfd_predicted = results['test_ppfd_predicted']
            r_squared = results['test_r_squared']
            title_prefix = "TESTING"
            color = 'green'

        # Scatter plot
        ax.scatter(ppfd_measured, ppfd_predicted, s=100, alpha=0.6, edgecolor='black', linewidth=1.5, color=color)

        # Identity line
        min_val = min(ppfd_measured.min(), ppfd_predicted.min())
        max_val = max(ppfd_measured.max(), ppfd_predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')

        ax.set_xlabel('Measured PPFD (µmol/m²/s)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted PPFD (µmol/m²/s)', fontsize=10, fontweight='bold')
        ax.set_title(f'{title_prefix}: Measured vs Predicted (R²={r_squared:.4f})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    def _plot_grid_coordinates(self, ax, results, mode='train'):
        """Plot measurement points in X-Y coordinates with M/P values"""
        if mode == 'train':
            x_coords = results['train_x_coords']
            y_coords = results['train_y_coords']
            ppfd_measured = results['train_ppfd_measured']
            ppfd_predicted = results['train_ppfd_predicted']
            residuals = results['train_residuals']
            title_prefix = "TRAINING"
        else:
            x_coords = results['test_x_coords']
            y_coords = results['test_y_coords']
            ppfd_measured = results['test_ppfd_measured']
            ppfd_predicted = results['test_ppfd_predicted']
            residuals = results['test_residuals']
            title_prefix = "TESTING"

        # Get unique X and Y for grid lines
        unique_x = sorted(set(x_coords))
        unique_y = sorted(set(y_coords))

        # Draw grid lines
        for y in unique_y:
            ax.axhline(y=y, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)
        for x in unique_x:
            ax.axvline(x=x, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Plot points with colors based on error (residuals already extracted above based on mode)
        scatter = ax.scatter(x_coords, y_coords, c=np.abs(residuals), cmap='YlOrRd',
                           s=150, edgecolor='black', linewidth=1.2, zorder=5)

        # Add text labels
        for i in range(len(x_coords)):
            ax.text(x_coords[i], y_coords[i], f'M:{ppfd_measured[i]:.0f}\nP:{ppfd_predicted[i]:.0f}',
                   ha='center', va='center', fontsize=6, fontweight='bold', color='black', zorder=6)

        ax.set_xlabel('X (cm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (cm)', fontsize=10, fontweight='bold')
        ax.set_title(f'{title_prefix}: Measurement Points (M=Measured, P=Predicted)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar(scatter, ax=ax, label='Absolute Error (µmol/m²/s)', shrink=0.8)

    def _plot_grid_search_sse(self, ax, results):
        """Plot SSE vs alpha for grid search"""
        grid_data = self.grid_search_data
        alpha_values = grid_data['alpha_values']
        sse_values = grid_data['sse_values']

        ax.plot(alpha_values, sse_values, 'g-', linewidth=2.5)
        ax.scatter([results['alpha']], [results['sse']], color='red', s=200, marker='*',
                  label=f"Optimal: α={results['alpha']:.6f}", zorder=5, edgecolor='darkred', linewidth=2)
        ax.set_xlabel('α (Decay Rate, cm⁻¹)', fontsize=10, fontweight='bold')
        ax.set_ylabel('SSE', fontsize=10, fontweight='bold')
        ax.set_title('Cost Function vs Alpha', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_grid_search_theta1(self, ax, results):
        """Plot theta1 vs alpha for grid search"""
        grid_data = self.grid_search_data
        alpha_values = grid_data['alpha_values']
        theta1_values = grid_data['theta1_values']

        ax.plot(alpha_values, theta1_values, 'b-', linewidth=2.5)
        ax.scatter([results['alpha']], [results['theta1']], color='red', s=200, marker='*',
                  label=f"Optimal: θ₁={results['theta1']:.6f}", zorder=5, edgecolor='darkred', linewidth=2)
        ax.set_xlabel('α (Decay Rate, cm⁻¹)', fontsize=10, fontweight='bold')
        ax.set_ylabel('θ₁ (µmol/m²/s)', fontsize=10, fontweight='bold')
        ax.set_title('Optimal θ₁ vs Alpha', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _create_detailed_tables(self, parent, results):
        """Create combined table for all training data and separate table for testing"""

        # Get data
        lamp_type = results['lamp_type']
        train_heights = results['train_heights']
        test_height = results['test_height']

        # Split training data by height
        train_data_by_height = self._split_training_data_by_height(results)

        # Create combined training table
        self._create_combined_training_table(parent, train_data_by_height, train_heights, lamp_type)

        # Create table for testing height (only if test_height is not None)
        if test_height is not None:
            test_data = {
                'x_coords': results['test_x_coords'],
                'y_coords': results['test_y_coords'],
                'measured': results['test_ppfd_measured'],
                'predicted': results['test_ppfd_predicted'],
                'idx': results['test_idx']
            }
            self._create_single_height_table(parent, test_data,
                                             test_height, lamp_type, 'Testing')

    def _create_combined_training_table(self, parent, train_data_by_height, train_heights, lamp_type):
        """Create a single combined table for all training data"""

        # Collect all data organized by IDX
        data_by_idx = {}
        for i, height in enumerate(train_heights):
            data = train_data_by_height[i]
            for j in range(len(data['idx'])):
                idx = int(data['idx'][j])
                if idx not in data_by_idx:
                    data_by_idx[idx] = {
                        'x': data['x_coords'][j],
                        'y': data['y_coords'][j],
                        'heights': {}
                    }
                data_by_idx[idx]['heights'][height] = {
                    'measured': data['measured'][j],
                    'predicted': data['predicted'][j]
                }

        # Build headers dynamically based on number of training heights
        headers = ['IDX', 'X', 'Y']
        for i, h in enumerate(train_heights):
            real_h = h + self.get_height_offset(lamp_type, h)
            headers.append(f'H={real_h}\nError')
            headers.append(f'H={real_h}\nErr%')

        # Build table data
        table_data = []
        for idx in sorted(data_by_idx.keys()):
            row_data = data_by_idx[idx]
            row = [
                f'{idx}',
                f'{row_data["x"]:.0f}',
                f'{row_data["y"]:.0f}'
            ]
            for height in train_heights:
                if height in row_data['heights']:
                    measured = row_data['heights'][height]['measured']
                    predicted = row_data['heights'][height]['predicted']
                    error = predicted - measured
                    error_pct = (error / measured) * 100 if measured != 0 else 0
                    row.append(f'{error:+.0f}')
                    row.append(f'{error_pct:+.1f}%')
                else:
                    row.append('-')
                    row.append('-')
            table_data.append(row)

        # Calculate optimal figure dimensions
        num_height_cols = len(train_heights) * 2
        total_cols = 3 + num_height_cols
        num_rows = len(table_data)

        # Width: based on columns (more columns = wider)
        fig_width = min(4 + total_cols * 0.9, 16)

        # Height: based on rows (more rows = taller), with limits
        row_height = 0.35  # Height per row
        header_space = 2.5  # Space for title and header
        fig_height = min(header_space + num_rows * row_height, 12)  # Max 12 inches

        # Create figure for table
        fig_table = Figure(figsize=(fig_width, fig_height), dpi=100)
        ax_table = fig_table.add_subplot(111)
        ax_table.axis('tight')
        ax_table.axis('off')

        # Calculate column widths dynamically
        base_width = 0.10  # IDX
        coord_width = 0.09  # X, Y
        height_col_width = 0.12  # Each measurement/prediction column

        col_widths = [base_width, coord_width, coord_width]
        col_widths.extend([height_col_width] * num_height_cols)

        # Create table
        table = ax_table.table(cellText=table_data, colLabels=headers,
                              cellLoc='center', loc='upper center',
                              colWidths=col_widths)

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)

        # Color headers
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2E5C8A')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=8)
            table[(0, i)].set_height(0.08)

        # Color rows - alternate
        for i in range(1, len(table_data) + 1):
            color = '#F5F5F5' if i % 2 == 0 else 'white'
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_height(0.05)

        # Add title
        title_text = f'Training Data - All Heights'
        ax_table.set_title(title_text, fontsize=11, fontweight='bold', pad=20)

        # Adjust layout to show everything
        fig_table.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)

        # Embed in tkinter
        canvas_table = FigureCanvasTkAgg(fig_table, master=parent)
        canvas_table.draw()
        canvas_table.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def _create_single_height_table(self, parent, data, nominal_height, lamp_type, title):
        """Create a matplotlib table for testing data"""

        x_coords = data['x_coords']
        y_coords = data['y_coords']
        measured = data['measured']
        predicted = data['predicted']
        idx_vals = data['idx']
        errors = predicted - measured
        error_pct = (errors / measured) * 100

        # Prepare table data
        table_data = []
        headers = ['IDX', 'X', 'Y', 'Meas', 'Pred', 'Error', 'Err %']

        for i in range(len(x_coords)):
            table_data.append([
                f'{int(idx_vals[i])}',
                f'{x_coords[i]:.0f}',
                f'{y_coords[i]:.0f}',
                f'{measured[i]:.0f}',
                f'{predicted[i]:.0f}',
                f'{errors[i]:+.0f}',
                f'{error_pct[i]:+.1f}%'
            ])

        # Calculate figure dimensions based on data
        num_rows = len(table_data)
        row_height = 0.35
        header_space = 2.5
        fig_height = min(header_space + num_rows * row_height, 12)
        fig_width = 9

        # Create figure for table
        fig_table = Figure(figsize=(fig_width, fig_height), dpi=100)
        ax_table = fig_table.add_subplot(111)
        ax_table.axis('tight')
        ax_table.axis('off')

        # Create table
        table = ax_table.table(cellText=table_data, colLabels=headers,
                              cellLoc='center', loc='upper center',
                              colWidths=[0.12, 0.11, 0.11, 0.14, 0.14, 0.14, 0.14])

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)

        # Color headers (green for testing)
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2E7D32')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=8)
            table[(0, i)].set_height(0.08)

        # Color rows - alternate
        for i in range(1, len(table_data) + 1):
            color = '#F5F5F5' if i % 2 == 0 else 'white'
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_height(0.05)

        # Add title
        height_offset = self.get_height_offset(lamp_type, nominal_height)
        real_height = nominal_height + height_offset
        title_text = f'{title} Data - Height: {real_height}cm'
        ax_table.set_title(title_text, fontsize=11, fontweight='bold', pad=20)

        # Adjust layout to show everything
        fig_table.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)

        # Embed in tkinter
        canvas_table = FigureCanvasTkAgg(fig_table, master=parent)
        canvas_table.draw()
        canvas_table.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def _create_error_analysis_table_matplotlib(self, parent, results):
        """Create error analysis table using matplotlib"""
        # Create figure for table
        fig_table = Figure(figsize=(14, 3), dpi=100)
        ax_table = fig_table.add_subplot(111)
        ax_table.axis('tight')
        ax_table.axis('off')

        # Get data
        lamp_type = results['lamp_type']
        train_heights = results['train_heights']
        test_height = results['test_height']

        # Split training data by height
        train_data_by_height = self._split_training_data_by_height(results)

        # Prepare table data
        table_data = []
        headers = ['Height Type', 'Nominal\n(cm)', 'Real\n(cm)', 'Points', 'Mean Error', 'Abs Mean\nError', 'RMSE', 'R²']

        # Add training heights
        for i, nominal_h in enumerate(train_heights):
            real_h = nominal_h + self.get_height_offset(lamp_type, nominal_h)
            data = train_data_by_height[i]
            errors = data['predicted'] - data['measured']

            mean_error = np.mean(errors)
            abs_mean_error = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))

            # Calculate R²
            ss_res = np.sum(errors**2)
            ss_tot = np.sum((data['measured'] - np.mean(data['measured']))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            table_data.append([
                f'Training {i+1}',
                f'{nominal_h:.1f}',
                f'{real_h:.1f}',
                f'{len(errors)}',
                f'{mean_error:+.2f}',
                f'{abs_mean_error:.2f}',
                f'{rmse:.2f}',
                f'{r_squared:.4f}'
            ])

        # Add testing height
        test_errors = results['test_ppfd_predicted'] - results['test_ppfd_measured']
        test_real_h = test_height + self.get_height_offset(lamp_type, test_height)
        mean_error_test = np.mean(test_errors)
        abs_mean_error_test = np.mean(np.abs(test_errors))

        table_data.append([
            'Testing',
            f'{test_height:.1f}',
            f'{test_real_h:.1f}',
            f'{results["n_test"]}',
            f'{mean_error_test:+.2f}',
            f'{abs_mean_error_test:.2f}',
            f'{results["test_rmse"]:.2f}',
            f'{results["test_r_squared"]:.4f}'
        ])

        # Create table
        table = ax_table.table(cellText=table_data, colLabels=headers,
                              cellLoc='center', loc='center',
                              colWidths=[0.15, 0.1, 0.1, 0.08, 0.12, 0.12, 0.1, 0.1])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Color coding
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color rows
        for i in range(1, len(table_data) + 1):
            if i <= 2:  # Training rows
                color = '#E3F2FD'
            else:  # Testing row
                color = '#E8F5E9'
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)

        # Add title
        ax_table.set_title('Error Analysis by Height', fontsize=14, fontweight='bold', pad=20)

        fig_table.tight_layout()

        # Embed in tkinter
        canvas_table = FigureCanvasTkAgg(fig_table, master=parent)
        canvas_table.draw()
        canvas_table.get_tk_widget().pack(fill='both', expand=True)

    def _create_error_analysis_table(self, parent, results):
        """Create error analysis table showing statistics by height"""
        # Create container
        container = ttk.Frame(parent)
        container.pack(fill='x', expand=False, padx=5, pady=5)

        # Style configuration for better visibility
        style = ttk.Style()
        style.configure("Error.Treeview", rowheight=30, font=('Arial', 10))
        style.configure("Error.Treeview.Heading", font=('Arial', 10, 'bold'))

        # Create treeview
        columns = ('Height Type', 'Nominal (cm)', 'Real (cm)', 'Points', 'Mean Error', 'Abs Mean Error', 'RMSE', 'R²')
        tree = ttk.Treeview(container, columns=columns, height=4, show='headings', style="Error.Treeview")

        # Define column widths
        widths = [120, 120, 100, 80, 120, 140, 100, 100]
        for col, width in zip(columns, widths):
            tree.column(col, width=width, anchor='center')
            tree.heading(col, text=col)

        # Get data
        lamp_type = results['lamp_type']
        train_heights = results['train_heights']
        test_height = results['test_height']

        # Split training data by height to calculate individual statistics
        train_data_by_height = self._split_training_data_by_height(results)

        # Add rows for each training height
        for i, nominal_h in enumerate(train_heights):
            real_h = nominal_h + self.get_height_offset(lamp_type, nominal_h)
            data = train_data_by_height[i]
            errors = data['predicted'] - data['measured']

            mean_error = np.mean(errors)
            abs_mean_error = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))

            # Calculate R² for this height
            ss_res = np.sum(errors**2)
            ss_tot = np.sum((data['measured'] - np.mean(data['measured']))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            tree.insert('', 'end', values=(
                f'Training {i+1}',
                f'{nominal_h:.1f}',
                f'{real_h:.1f}',
                f'{len(errors)}',
                f'{mean_error:+.2f}',
                f'{abs_mean_error:.2f}',
                f'{rmse:.2f}',
                f'{r_squared:.4f}'
            ), tags=('train',))

        # Add row for testing height
        test_errors = results['test_ppfd_predicted'] - results['test_ppfd_measured']
        test_real_h = test_height + self.get_height_offset(lamp_type, test_height)

        mean_error_test = np.mean(test_errors)
        abs_mean_error_test = np.mean(np.abs(test_errors))

        tree.insert('', 'end', values=(
            'Testing',
            f'{test_height:.1f}',
            f'{test_real_h:.1f}',
            f'{results["n_test"]}',
            f'{mean_error_test:+.2f}',
            f'{abs_mean_error_test:.2f}',
            f'{results["test_rmse"]:.2f}',
            f'{results["test_r_squared"]:.4f}'
        ), tags=('test',))

        # Color coding
        tree.tag_configure('train', background='#E3F2FD')
        tree.tag_configure('test', background='#E8F5E9')

        # Pack tree directly (no scrollbars needed for 3 rows)
        tree.pack(fill='x', expand=False, padx=5, pady=5)

    def _create_estimation_table(self, parent, results):
        """Create detailed results table for both training and testing"""
        # Create notebook for train/test tabs
        table_notebook = ttk.Notebook(parent)
        table_notebook.pack(fill='both', expand=True)

        # Training table
        train_frame = ttk.Frame(table_notebook)
        table_notebook.add(train_frame, text='Training Data')
        self._create_data_table(train_frame, results, 'train')

        # Testing table
        test_frame = ttk.Frame(table_notebook)
        table_notebook.add(test_frame, text='Testing Data')
        self._create_data_table(test_frame, results, 'test')

    def _create_data_table(self, parent, results, mode='train'):
        """Create a single data table for train or test"""
        if mode == 'train':
            ppfd_measured = results['train_ppfd_measured']
            ppfd_predicted = results['train_ppfd_predicted']
            residuals = results['train_residuals']
            x_coords = results['train_x_coords']
            y_coords = results['train_y_coords']
            lamp_heights = results['train_lamp_heights']
            n_points = results['n_train']
        else:
            ppfd_measured = results['test_ppfd_measured']
            ppfd_predicted = results['test_ppfd_predicted']
            residuals = results['test_residuals']
            x_coords = results['test_x_coords']
            y_coords = results['test_y_coords']
            lamp_heights = results['test_lamp_heights']
            n_points = results['n_test']

        # Create treeview with scrollbar
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill='both', expand=True, padx=5, pady=5)

        columns = ('P#', 'X (cm)', 'Y (cm)', 'Lamp H (cm)', 'Measured', 'Predicted', 'Error', 'Error %')
        tree = ttk.Treeview(tree_frame, columns=columns, height=15, show='headings')

        # Define column widths
        widths = [40, 80, 80, 100, 120, 120, 120, 100]
        for col, width in zip(columns, widths):
            tree.column(col, width=width, anchor='center')
            tree.heading(col, text=col)

        # Add data rows
        for i in range(n_points):
            error_pct = (residuals[i] / ppfd_measured[i] * 100) if ppfd_measured[i] != 0 else 0
            tree.insert('', 'end', values=(
                f'{i+1}',
                f'{x_coords[i]:.2f}',
                f'{y_coords[i]:.2f}',
                f'{lamp_heights[i]:.2f}',
                f'{ppfd_measured[i]:.2f}',
                f'{ppfd_predicted[i]:.2f}',
                f'{residuals[i]:+.2f}',
                f'{error_pct:+.2f}%'
            ))

        # Add scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient='horizontal', command=tree.xview)
        tree.configure(yscroll=vsb.set, xscroll=hsb.set)

        # Grid layout
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

    def create_ann_estimation_interface(self):
        """Create the ANN estimation tab interface"""

        # Left panel container with scrollbar for controls
        left_container = ttk.Frame(self.ann_estimation_frame)
        left_container.pack(side='left', fill='both', padx=10, pady=10, expand=False)

        # Canvas and scrollbar for left panel
        left_canvas = tk.Canvas(left_container, width=380)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        self.left_panel_ann = ttk.Frame(left_canvas)

        self.left_panel_ann.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )

        left_canvas.create_window((0, 0), window=self.left_panel_ann, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling for left panel
        def _on_left_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel)

        # Title
        title_label = ttk.Label(self.left_panel_ann,
                               text="ANN Parameter Estimation",
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Right panel for results (same structure as Linear Regression)
        self.right_panel_ann = ttk.Frame(self.ann_estimation_frame)
        self.right_panel_ann.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        # === LEFT PANEL: Controls ===

        # 1. Lamp Configuration
        lamp_config_frame = ttk.LabelFrame(self.left_panel_ann, text="Lamp Configuration", padding=10)
        lamp_config_frame.pack(fill='x', pady=5)

        ttk.Label(lamp_config_frame, text="LED Type:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=5)
        self.ann_lamp_type = tk.StringVar(value='Quantum')
        lamp_combo = ttk.Combobox(lamp_config_frame, textvariable=self.ann_lamp_type,
                                  values=['Quantum', '12V', '54V'], state='readonly', width=15)
        lamp_combo.grid(row=0, column=1, padx=5, pady=5)
        lamp_combo.bind('<<ComboboxSelected>>', lambda e: self.update_ann_lamp_specs_display())

        self.ann_lamp_specs_label = ttk.Label(lamp_config_frame, text="", font=('Arial', 9))
        self.ann_lamp_specs_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=5)

        ttk.Label(lamp_config_frame, text="Orientation:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=5)
        self.ann_orientation = tk.StringVar(value='perpendicular')
        ttk.Radiobutton(lamp_config_frame, text="Perpendicular (X-axis)",
                       variable=self.ann_orientation, value='perpendicular').grid(row=2, column=1, sticky='w')
        ttk.Radiobutton(lamp_config_frame, text="Parallel (Y-axis)",
                       variable=self.ann_orientation, value='parallel').grid(row=3, column=1, sticky='w')

        # 2. Data Status
        data_status_frame = ttk.LabelFrame(self.left_panel_ann, text="Data Status", padding=10)
        data_status_frame.pack(fill='x', pady=5)

        self.ann_data_status_label = ttk.Label(data_status_frame, text="No data loaded",
                                               font=('Arial', 9), foreground='orange')
        self.ann_data_status_label.pack(anchor='w')

        # 3. Height Selection
        height_selection_frame = ttk.LabelFrame(self.left_panel_ann, text="Height Selection", padding=10)
        height_selection_frame.pack(fill='x', pady=5)

        ttk.Label(height_selection_frame, text="Training Heights:",
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky='w', pady=5)

        for i in range(3):
            ttk.Label(height_selection_frame, text=f"Height {i+1}:").grid(row=i+1, column=0, sticky='w', padx=5)
            setattr(self, f'ann_train_height{i+1}', tk.StringVar(value='None'))
            combo = ttk.Combobox(height_selection_frame,
                               textvariable=getattr(self, f'ann_train_height{i+1}'),
                               values=['None'], state='readonly', width=18)
            combo.grid(row=i+1, column=1, padx=5, pady=2)
            setattr(self, f'ann_train_height{i+1}_combo', combo)

        ttk.Label(height_selection_frame, text="Testing Height:",
                 font=('Arial', 10, 'bold')).grid(row=4, column=0, columnspan=2, sticky='w', pady=(10,5))
        ttk.Label(height_selection_frame, text="Height:").grid(row=5, column=0, sticky='w', padx=5)
        self.ann_test_height = tk.StringVar(value='None')
        self.ann_test_height_combo = ttk.Combobox(height_selection_frame,
                                                  textvariable=self.ann_test_height,
                                                  values=['None'], state='readonly', width=18)
        self.ann_test_height_combo.grid(row=5, column=1, padx=5, pady=2)

        self.ann_height_note_label = ttk.Label(height_selection_frame, text="",
                                              font=('Arial', 8, 'italic'))
        self.ann_height_note_label.grid(row=6, column=0, columnspan=2, sticky='w', pady=5)

        # 4. ANN Architecture
        ann_arch_frame = ttk.LabelFrame(self.left_panel_ann, text="ANN Architecture", padding=10)
        ann_arch_frame.pack(fill='x', pady=5)

        ttk.Label(ann_arch_frame, text="Number of Hidden Layers:",
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=5)
        self.ann_num_layers = tk.IntVar(value=3)
        ttk.Spinbox(ann_arch_frame, from_=1, to=5, textvariable=self.ann_num_layers,
                   width=10, command=self.update_ann_layer_inputs).grid(row=0, column=1, padx=5)

        # Container for layer neuron inputs
        self.ann_layers_container = ttk.Frame(ann_arch_frame)
        self.ann_layers_container.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)

        ttk.Label(ann_arch_frame, text="Activation Function:",
                 font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=5)
        self.ann_activation = tk.StringVar(value='relu')
        ttk.Combobox(ann_arch_frame, textvariable=self.ann_activation,
                    values=['relu', 'tanh', 'sigmoid', 'elu'],
                    state='readonly', width=10).grid(row=2, column=1, padx=5)

        ttk.Label(ann_arch_frame, text="Learning Rate:",
                 font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky='w', pady=5)
        self.ann_learning_rate = tk.DoubleVar(value=0.001)
        ttk.Entry(ann_arch_frame, textvariable=self.ann_learning_rate,
                 width=12).grid(row=3, column=1, padx=5)

        ttk.Label(ann_arch_frame, text="Epochs:",
                 font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky='w', pady=5)
        self.ann_epochs = tk.IntVar(value=100)
        ttk.Spinbox(ann_arch_frame, from_=10, to=1000, textvariable=self.ann_epochs,
                   width=10, increment=10).grid(row=4, column=1, padx=5)

        ttk.Label(ann_arch_frame, text="Batch Size:",
                 font=('Arial', 10, 'bold')).grid(row=5, column=0, sticky='w', pady=5)
        self.ann_batch_size = tk.IntVar(value=16)
        ttk.Spinbox(ann_arch_frame, from_=4, to=64, textvariable=self.ann_batch_size,
                   width=10, increment=4).grid(row=5, column=1, padx=5)

        # Initialize layer inputs
        self.update_ann_layer_inputs()

        # 5. Training Progress
        progress_frame = ttk.LabelFrame(self.left_panel_ann, text="Training Progress", padding=10)
        progress_frame.pack(fill='x', pady=5)

        self.ann_progress_var = tk.DoubleVar(value=0)
        self.ann_progress_bar = ttk.Progressbar(progress_frame, variable=self.ann_progress_var,
                                               maximum=100, length=200)
        self.ann_progress_bar.pack(fill='x', pady=5)

        self.ann_progress_label = ttk.Label(progress_frame, text="Ready",
                                           font=('Arial', 9), foreground='gray')
        self.ann_progress_label.pack()

        # 6. Run Button
        button_frame = ttk.Frame(self.left_panel_ann)
        button_frame.pack(fill='x', pady=10)

        self.ann_run_button = ttk.Button(button_frame, text="Run ANN Estimation",
                                         command=self.run_ann_estimation)
        self.ann_run_button.pack(fill='x', pady=5)

        # Initialize displays
        self.update_ann_lamp_specs_display()
        self.update_ann_estimation_data_status()

    def update_ann_layer_inputs(self):
        """Update layer neuron input fields based on number of layers"""
        # Clear existing inputs
        for widget in self.ann_layers_container.winfo_children():
            widget.destroy()

        # Create new inputs
        num_layers = self.ann_num_layers.get()
        self.ann_layer_neurons = []

        default_neurons = [64, 32, 16, 8, 4]  # Default neuron counts

        for i in range(num_layers):
            ttk.Label(self.ann_layers_container,
                     text=f"Layer {i+1} Neurons:").grid(row=i, column=0, sticky='w', pady=2)

            neuron_var = tk.IntVar(value=default_neurons[i] if i < len(default_neurons) else 16)
            self.ann_layer_neurons.append(neuron_var)

            ttk.Spinbox(self.ann_layers_container, from_=4, to=256,
                       textvariable=neuron_var, width=10, increment=4).grid(row=i, column=1, padx=5, pady=2)

    def update_ann_lamp_specs_display(self):
        """Update lamp specifications display for ANN tab"""
        lamp_type = self.ann_lamp_type.get()
        specs = self.led_specs[lamp_type]
        text = f"LEDs: {specs['num_leds']} | Length: {specs['length']}cm | Cost: {specs['cost']} TL"
        if lamp_type == 'Quantum':
            text += f" | Height offset: 13cm→0cm, 27/35cm→+4cm"
        self.ann_lamp_specs_label.config(text=text)

        # Update data status
        self.update_ann_estimation_data_status()

    def update_ann_estimation_data_status(self):
        """Update data status in ANN estimation tab"""
        if not hasattr(self, 'ann_data_status_label'):
            return

        if self.lamp_data and self.idx_points_df is not None:
            lamp_type = self.ann_lamp_type.get()
            real_heights = [h + self.get_height_offset(lamp_type, h) for h in self.available_heights]

            text = f"Data loaded: {len(self.idx_points_df)} points, {len(self.available_heights)} heights\n"
            text += f"Nominal Heights: {self.available_heights}\n"

            if lamp_type == 'Quantum':
                text += f"Real Heights ({lamp_type}): {real_heights} (13cm→0, 27/35cm→+4cm)"
            else:
                text += f"Real Heights ({lamp_type}): {real_heights}"

            self.ann_data_status_label.config(text=text, foreground='green')

            # Populate height selection comboboxes
            height_options = [f"{h}cm (Real: {h + self.get_height_offset(lamp_type, h)}cm)"
                            if lamp_type == 'Quantum' else f"{h}cm"
                            for h in self.available_heights]

            self.ann_train_height1_combo['values'] = ['None'] + height_options
            self.ann_train_height2_combo['values'] = ['None'] + height_options
            self.ann_train_height3_combo['values'] = ['None'] + height_options
            self.ann_test_height_combo['values'] = ['None'] + height_options

            # Set defaults
            if len(height_options) >= 2:
                self.ann_train_height1.set(height_options[0])
                self.ann_train_height2.set(height_options[1])
            if len(height_options) >= 3:
                self.ann_test_height.set(height_options[2])

            # Update note
            if lamp_type == 'Quantum':
                self.ann_height_note_label.config(
                    text=f"Note: For {lamp_type}, 13cm→real=13cm, 27cm→real=31cm, 35cm→real=39cm",
                    foreground='blue')
            else:
                self.ann_height_note_label.config(text="")
        else:
            self.ann_data_status_label.config(
                text="No data loaded. Please load data from Visualization tab first.",
                foreground='orange')

    def run_ann_estimation(self):
        """Run ANN parameter estimation"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            messagebox.showerror("Error", f"Required library not found: {str(e)}\n\nPlease install: pip install tensorflow scikit-learn")
            return

        # Check if data is loaded
        if not self.lamp_data or self.idx_points_df is None:
            messagebox.showerror("Error", "Please load data from the Visualization tab first!")
            return

        # Get lamp configuration
        lamp_type = self.ann_lamp_type.get()
        orientation = self.ann_orientation.get()

        # Parse selected heights
        def parse_height(height_str):
            if height_str == 'None' or not height_str:
                return None
            return int(height_str.split('cm')[0])

        train_heights = []
        for i in range(1, 4):
            h = parse_height(getattr(self, f'ann_train_height{i}').get())
            if h is not None:
                train_heights.append(h)

        test_height = parse_height(self.ann_test_height.get())

        if len(train_heights) == 0:
            messagebox.showerror("Error", "Please select at least one Training Height!")
            return

        # Get ANN parameters
        num_layers = self.ann_num_layers.get()
        layer_neurons = [var.get() for var in self.ann_layer_neurons]
        activation = self.ann_activation.get()
        learning_rate = self.ann_learning_rate.get()
        epochs = self.ann_epochs.get()
        batch_size = self.ann_batch_size.get()

        # Show confirmation
        train_real_heights = [h + self.get_height_offset(lamp_type, h) for h in train_heights]
        info_msg = f"Lamp Type: {lamp_type}\n"
        info_msg += f"Orientation: {orientation}\n\n"
        info_msg += f"Training Heights: {train_heights} cm (Real: {train_real_heights} cm)\n"
        if test_height:
            test_real = test_height + self.get_height_offset(lamp_type, test_height)
            info_msg += f"Testing Height: {test_height} cm (Real: {test_real} cm)\n\n"
        info_msg += f"ANN Architecture: 3 → {' → '.join(map(str, layer_neurons))} → 1\n"
        info_msg += f"Activation: {activation}, LR: {learning_rate}, Epochs: {epochs}"

        messagebox.showinfo("ANN Estimation Setup", info_msg)

        # Prepare training data
        X_train_list = []
        y_train_list = []
        train_idx_list = []

        for height in train_heights:
            lamp_df = self.lamp_data[height]
            merged_df = pd.merge(lamp_df, self.idx_points_df, on='idx', how='inner')

            x_coords = merged_df['X'].values
            y_coords = merged_df['Y'].values
            ppfd_values = merged_df[lamp_type].values * 0.8
            idx_values = merged_df['idx'].values
            real_height = height + self.get_height_offset(lamp_type, height)

            for x, y, ppfd, idx in zip(x_coords, y_coords, ppfd_values, idx_values):
                X_train_list.append([x, y, real_height])
                y_train_list.append(ppfd)
                train_idx_list.append(idx)

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        train_idx = np.array(train_idx_list)

        # Prepare testing data
        X_test = None
        y_test = None
        test_idx = None
        if test_height:
            X_test_list = []
            y_test_list = []
            test_idx_list = []

            test_lamp_df = self.lamp_data[test_height]
            test_merged = pd.merge(test_lamp_df, self.idx_points_df, on='idx', how='inner')

            x_coords = test_merged['X'].values
            y_coords = test_merged['Y'].values
            ppfd_values = test_merged[lamp_type].values * 0.8
            idx_values = test_merged['idx'].values
            test_real_height = test_height + self.get_height_offset(lamp_type, test_height)

            for x, y, ppfd, idx in zip(x_coords, y_coords, ppfd_values, idx_values):
                X_test_list.append([x, y, test_real_height])
                y_test_list.append(ppfd)
                test_idx_list.append(idx)

            X_test = np.array(X_test_list)
            y_test = np.array(y_test_list)
            test_idx = np.array(test_idx_list)

        # Normalize data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Create model
        model = keras.Sequential()
        model.add(layers.Dense(layer_neurons[0], activation=activation, input_shape=(3,)))

        for i in range(1, num_layers):
            model.add(layers.Dense(layer_neurons[i], activation=activation))

        model.add(layers.Dense(1, activation='linear'))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        # Reset progress
        self.ann_progress_var.set(0)
        self.ann_progress_label.config(text="Training in progress...", foreground='blue')
        self.root.update()

        # Custom callback for progress
        class ProgressCallback(keras.callbacks.Callback):
            def __init__(self, gui, total_epochs):
                super().__init__()
                self.gui = gui
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                progress = ((epoch + 1) / self.total_epochs) * 100
                self.gui.ann_progress_var.set(progress)
                self.gui.ann_progress_label.config(
                    text=f"Epoch {epoch+1}/{self.total_epochs} - Loss: {logs['loss']:.4f}",
                    foreground='blue')
                self.gui.root.update()

        # Train model
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[ProgressCallback(self, epochs)]
        )

        # Make predictions on training data
        y_train_pred_scaled = model.predict(X_train_scaled, verbose=0).flatten()
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()

        # Prepare results dictionary
        results = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'lamp_type': lamp_type,
            'orientation': orientation,
            'train_heights': train_heights,
            'test_height': test_height,
            'num_layers': num_layers,
            'layer_neurons': layer_neurons,
            'activation': activation,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'X_train': X_train,
            'y_train': y_train,
            'y_train_pred': y_train_pred,
            'train_x_coords': X_train[:, 0],
            'train_y_coords': X_train[:, 1],
            'train_lamp_heights': X_train[:, 2],
            'train_idx': train_idx,
            'n_train': len(X_train),
            'history': history
        }

        # Add test data if exists
        if X_test is not None:
            X_test_scaled = scaler_X.transform(X_test)
            y_test_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
            y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

            results['X_test'] = X_test
            results['y_test'] = y_test
            results['y_test_pred'] = y_test_pred
            results['test_x_coords'] = X_test[:, 0]
            results['test_y_coords'] = X_test[:, 1]
            results['test_lamp_heights'] = X_test[:, 2]
            results['test_idx'] = test_idx
            results['n_test'] = len(X_test)

        # Update progress
        self.ann_progress_var.set(100)
        self.ann_progress_label.config(text="Training completed!", foreground='green')

        # Display results
        self.display_ann_results(results)

        messagebox.showinfo("Success", "ANN estimation completed successfully!")

    def display_ann_results(self, results):
        """Display ANN estimation results with detailed visualizations"""
        # Clear right panel
        for widget in self.right_panel_ann.winfo_children():
            widget.destroy()

        # Create canvas with scrollbar (same as Linear Regression)
        canvas = tk.Canvas(self.right_panel_ann, bg='white')
        scrollbar = ttk.Scrollbar(self.right_panel_ann, orient="vertical", command=canvas.yview)

        # Create a frame inside the canvas
        scrollable_frame = ttk.Frame(canvas)

        # Configure the canvas
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Main container
        main_container = scrollable_frame

        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(title_frame, text="ANN Estimation Results",
                 font=('Arial', 14, 'bold')).pack()

        # Top section: Summary
        summary_frame = ttk.LabelFrame(main_container, text="Summary", padding=10)
        summary_frame.pack(fill='x', padx=10, pady=5)

        # Left and right columns
        left_summary = ttk.Frame(summary_frame)
        left_summary.pack(side='left', fill='both', expand=True, padx=10)

        right_summary = ttk.Frame(summary_frame)
        right_summary.pack(side='right', fill='both', expand=True, padx=10)

        # Left column: Model Architecture
        lamp_type = results['lamp_type']
        train_real_heights = [h + self.get_height_offset(lamp_type, h) for h in results['train_heights']]
        test_height = results['test_height']

        ttk.Label(left_summary, text=f"Lamp Type: {lamp_type}", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(left_summary, text=f"Orientation: {results['orientation'].upper()}", font=('Arial', 10)).pack(anchor='w', pady=2)

        ttk.Label(left_summary, text=f"Training Heights (nominal): {results['train_heights']} cm", font=('Arial', 9)).pack(anchor='w', pady=1)
        if lamp_type == 'Quantum':
            ttk.Label(left_summary, text=f"Training Heights (real): {train_real_heights} cm", font=('Arial', 9, 'bold'), foreground='blue').pack(anchor='w', pady=1)

        if test_height is not None:
            ttk.Label(left_summary, text=f"Testing Height (nominal): {test_height} cm", font=('Arial', 9)).pack(anchor='w', pady=1)
            if lamp_type == 'Quantum':
                test_real_height = test_height + self.get_height_offset(lamp_type, test_height)
                ttk.Label(left_summary, text=f"Testing Height (real): {test_real_height} cm", font=('Arial', 9, 'bold'), foreground='blue').pack(anchor='w', pady=1)

        ttk.Label(left_summary, text="", font=('Arial', 2)).pack(anchor='w', pady=1)

        # Model architecture
        layer_str = ' → '.join(map(str, results['layer_neurons']))
        ttk.Label(left_summary, text=f"Model Architecture:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(left_summary, text=f"  3 (Input) → {layer_str} → 1 (Output)", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(left_summary, text=f"  Activation: {results['activation']}", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(left_summary, text=f"  Learning Rate: {results['learning_rate']}", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(left_summary, text=f"  Epochs: {results['epochs']}, Batch Size: {results['batch_size']}", font=('Arial', 9)).pack(anchor='w')

        # Right column: Performance Metrics
        y_train = results['y_train']
        y_train_pred = results['y_train_pred']
        train_r2, train_rmse, train_mae = self.calculate_metrics(y_train, y_train_pred)

        ttk.Label(right_summary, text="Training Performance:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(right_summary, text=f"  R²: {train_r2:.6f}", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(right_summary, text=f"  RMSE: {train_rmse:.4f} µmol/m²/s", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(right_summary, text=f"  MAE: {train_mae:.4f} µmol/m²/s", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(right_summary, text=f"  Samples: {results['n_train']}", font=('Arial', 9)).pack(anchor='w')

        if test_height is not None:
            y_test = results['y_test']
            y_test_pred = results['y_test_pred']
            test_r2, test_rmse, test_mae = self.calculate_metrics(y_test, y_test_pred)

            ttk.Label(right_summary, text="", font=('Arial', 2)).pack(anchor='w', pady=1)
            ttk.Label(right_summary, text="Testing Performance:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
            ttk.Label(right_summary, text=f"  R²: {test_r2:.6f}", font=('Arial', 9)).pack(anchor='w')
            ttk.Label(right_summary, text=f"  RMSE: {test_rmse:.4f} µmol/m²/s", font=('Arial', 9)).pack(anchor='w')
            ttk.Label(right_summary, text=f"  MAE: {test_mae:.4f} µmol/m²/s", font=('Arial', 9)).pack(anchor='w')
            ttk.Label(right_summary, text=f"  Samples: {results['n_test']}", font=('Arial', 9)).pack(anchor='w')

        # Graphs section
        graph_frame = ttk.Frame(main_container)
        graph_frame.pack(fill='x', padx=10, pady=10)

        # Split training data by height and create plots
        self._create_ann_plots(graph_frame, results)

        # Interactive Prediction section
        interactive_frame = ttk.LabelFrame(main_container, text="Interactive Prediction Calculator", padding=10)
        interactive_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(interactive_frame, text="Enter coordinates to predict PPFD:",
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=6, pady=5)

        ttk.Label(interactive_frame, text="X (cm):").grid(row=1, column=0, padx=5, pady=5)
        x_entry = ttk.Entry(interactive_frame, width=10)
        x_entry.grid(row=1, column=1, padx=5, pady=5)
        x_entry.insert(0, "0")

        ttk.Label(interactive_frame, text="Y (cm):").grid(row=1, column=2, padx=5, pady=5)
        y_entry = ttk.Entry(interactive_frame, width=10)
        y_entry.grid(row=1, column=3, padx=5, pady=5)
        y_entry.insert(0, "0")

        ttk.Label(interactive_frame, text="H (cm):").grid(row=1, column=4, padx=5, pady=5)
        h_entry = ttk.Entry(interactive_frame, width=10)
        h_entry.grid(row=1, column=5, padx=5, pady=5)
        first_train_height = results['train_heights'][0]
        default_h = first_train_height + self.get_height_offset(lamp_type, first_train_height)
        h_entry.insert(0, str(default_h))

        result_label = ttk.Label(interactive_frame, text="Predicted PPFD: --",
                                font=('Arial', 12, 'bold'), foreground='darkgreen',
                                background='lightyellow', relief='solid', padding=10)
        result_label.grid(row=2, column=0, columnspan=6, pady=10, sticky='ew')

        def predict_ppfd():
            try:
                x = float(x_entry.get())
                y = float(y_entry.get())
                h = float(h_entry.get())

                # Predict
                X_new = np.array([[x, y, h]])
                X_new_scaled = results['scaler_X'].transform(X_new)
                y_pred_scaled = results['model'].predict(X_new_scaled, verbose=0).flatten()
                y_pred = results['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]

                result_label.config(text=f"Predicted PPFD: {y_pred:.2f} µmol/m²/s")
            except Exception as e:
                result_label.config(text=f"Error: {str(e)}")

        ttk.Button(interactive_frame, text="Calculate", command=predict_ppfd).grid(row=3, column=0, columnspan=6, pady=5)

        note_text = "Note: Enter X, Y coordinates and lamp height (H) in cm. The model will predict PPFD value at that point."
        ttk.Label(interactive_frame, text=note_text,
                 font=('Arial', 8, 'italic'), foreground='gray',
                 wraplength=600).grid(row=4, column=0, columnspan=6, pady=5)

        # Additional section: IDX vs Predicted/Measured line plots (same as Linear Regression)
        line_plot_frame = ttk.Frame(main_container)
        line_plot_frame.pack(fill='x', padx=10, pady=10)

        # Get lamp_type and train_heights from results
        lamp_type_for_plots = results['lamp_type']
        train_heights_for_plots = results['train_heights']

        # Split training data by height
        train_data_by_height = []
        unique_train_heights = np.unique(results['train_lamp_heights'])

        for h in unique_train_heights:
            mask = results['train_lamp_heights'] == h
            train_data_by_height.append({
                'x_coords': results['train_x_coords'][mask],
                'y_coords': results['train_y_coords'][mask],
                'measured': results['y_train'][mask],
                'predicted': results['y_train_pred'][mask],
                'idx': np.where(mask)[0]  # Get indices
            })

        self._create_idx_line_plots(line_plot_frame, results, train_data_by_height, train_heights_for_plots, lamp_type_for_plots)

        # Bottom section: Detailed Tables for each height (same as Linear Regression)
        table_frame = ttk.Frame(main_container)
        table_frame.pack(fill='x', padx=10, pady=10)

        self._create_ann_detailed_tables(table_frame, results)

    def _create_ann_plots(self, parent, results):
        """Create plots for ANN results (similar to linear regression)"""
        lamp_type = results['lamp_type']
        train_heights = results['train_heights']
        test_height = results['test_height']

        # Split training data by height
        train_data_by_height = []
        unique_train_heights = np.unique(results['train_lamp_heights'])

        for h in unique_train_heights:
            mask = results['train_lamp_heights'] == h
            train_data_by_height.append({
                'x_coords': results['train_x_coords'][mask],
                'y_coords': results['train_y_coords'][mask],
                'measured': results['y_train'][mask],
                'predicted': results['y_train_pred'][mask]
            })

        # Prepare plot data
        all_plot_data = []
        for i, data in enumerate(train_data_by_height):
            # Find corresponding nominal height
            nominal_h = train_heights[i] if i < len(train_heights) else None
            all_plot_data.append({
                'data': data,
                'nominal_height': nominal_h,
                'title': f'Training {i+1}'
            })

        # Add test data if exists
        if test_height is not None:
            test_data = {
                'x_coords': results['test_x_coords'],
                'y_coords': results['test_y_coords'],
                'measured': results['y_test'],
                'predicted': results['y_test_pred']
            }
            all_plot_data.append({
                'data': test_data,
                'nominal_height': test_height,
                'title': 'Testing'
            })

        # Create plots (max 2 per row)
        num_total_plots = len(all_plot_data)
        plots_per_row = 2
        num_rows = (num_total_plots + plots_per_row - 1) // plots_per_row

        for row in range(num_rows):
            plots_in_this_row = min(plots_per_row, num_total_plots - row * plots_per_row)
            fig = Figure(figsize=(16, 6), dpi=90)

            for col in range(plots_in_this_row):
                plot_idx = row * plots_per_row + col
                plot_info = all_plot_data[plot_idx]

                ax = fig.add_subplot(1, plots_per_row, col + 1)
                self._plot_ann_comparison(ax, plot_info['data'], plot_info['nominal_height'],
                                         lamp_type, plot_info['title'])

            fig.tight_layout()

            fig_canvas = FigureCanvasTkAgg(fig, master=parent)
            fig_canvas.draw()
            fig_canvas.get_tk_widget().pack(fill='x', expand=False, pady=5)

    def _plot_ann_comparison(self, ax, data, nominal_height, lamp_type, data_type):
        """Plot measured vs predicted for ANN (grid style)"""
        x_coords = data['x_coords']
        y_coords = data['y_coords']
        measured = data['measured']
        predicted = data['predicted']

        # Get unique X and Y for grid lines
        unique_x = sorted(set(x_coords))
        unique_y = sorted(set(y_coords))

        # Draw grid lines
        for y in unique_y:
            ax.axhline(y=y, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)
        for x in unique_x:
            ax.axvline(x=x, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Calculate errors (percentage)
        errors = predicted - measured
        abs_errors = np.abs(errors)
        # Calculate percentage error (avoid division by zero)
        error_percentage = np.where(measured != 0, (abs_errors / measured) * 100, 0)

        # Plot points
        scatter = ax.scatter(x_coords, y_coords, c=error_percentage, cmap='YlGn',
                           s=180, edgecolor='black', linewidth=1.5, zorder=5)

        # Add text labels (positioned above the points)
        for i in range(len(x_coords)):
            ax.text(x_coords[i], y_coords[i] + 1.5, f'M:{measured[i]:.0f}\nP:{predicted[i]:.0f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold', color='black',
                   zorder=6)

        # Title
        height_offset = self.get_height_offset(lamp_type, nominal_height)
        real_height = nominal_height + height_offset
        if lamp_type == 'Quantum':
            title = f'{data_type}: {nominal_height}cm (Real: {real_height}cm)'
        else:
            title = f'{data_type}: {nominal_height}cm'

        ax.set_xlabel('X (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (cm)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        # Colorbar
        plt.colorbar(scatter, ax=ax, label='Error %')

    def _create_ann_detailed_tables(self, parent, results):
        """Create combined table for all ANN training data and separate table for testing"""

        # Get data
        lamp_type = results['lamp_type']
        train_heights = results['train_heights']
        test_height = results['test_height']

        # Split training data by height using actual idx values
        train_data_by_height = []
        unique_train_heights = np.unique(results['train_lamp_heights'])

        for h in unique_train_heights:
            mask = results['train_lamp_heights'] == h
            train_data_by_height.append({
                'x_coords': results['train_x_coords'][mask],
                'y_coords': results['train_y_coords'][mask],
                'measured': results['y_train'][mask],
                'predicted': results['y_train_pred'][mask],
                'idx': results['train_idx'][mask]  # Use actual idx from data
            })

        # Create combined training table
        self._create_combined_training_table(parent, train_data_by_height, train_heights, lamp_type)

        # Create table for testing height (only if test_height is not None)
        if test_height is not None:
            test_data = {
                'x_coords': results['test_x_coords'],
                'y_coords': results['test_y_coords'],
                'measured': results['y_test'],
                'predicted': results['y_test_pred'],
                'idx': results['test_idx']  # Use actual idx from data
            }
            self._create_single_height_table(parent, test_data,
                                             test_height, lamp_type, 'Testing')

    def calculate_metrics(self, y_true, y_pred):
        """Calculate R², RMSE, MAE"""
        residuals = y_true - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(ss_res / len(y_true))
        mae = np.mean(np.abs(residuals))
        return r_squared, rmse, mae

    # ========================== PINN ESTIMATION METHODS ==========================

    def create_pinn_estimation_interface(self):
        """Create the PINN estimation tab interface"""

        # Left panel container with scrollbar for controls
        left_container = ttk.Frame(self.pinn_estimation_frame)
        left_container.pack(side='left', fill='both', padx=10, pady=10, expand=False)

        # Canvas and scrollbar for left panel
        left_canvas = tk.Canvas(left_container, width=380)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        self.left_panel_pinn = ttk.Frame(left_canvas)

        self.left_panel_pinn.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )

        left_canvas.create_window((0, 0), window=self.left_panel_pinn, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling for left panel
        def _on_left_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel)

        # Title
        title_label = ttk.Label(self.left_panel_pinn,
                               text="PINN Parameter Estimation",
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)

        # Right panel for results (same structure as Linear Regression)
        self.right_panel_pinn = ttk.Frame(self.pinn_estimation_frame)
        self.right_panel_pinn.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        # === LEFT PANEL: Controls ===

        # 1. Lamp Configuration
        lamp_config_frame = ttk.LabelFrame(self.left_panel_pinn, text="Lamp Configuration", padding=10)
        lamp_config_frame.pack(fill='x', pady=5)

        ttk.Label(lamp_config_frame, text="LED Type:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=5)
        self.pinn_lamp_type = tk.StringVar(value='Quantum')
        lamp_combo = ttk.Combobox(lamp_config_frame, textvariable=self.pinn_lamp_type,
                                  values=['Quantum', '12V', '54V'], state='readonly', width=15)
        lamp_combo.grid(row=0, column=1, padx=5, pady=5)
        lamp_combo.bind('<<ComboboxSelected>>', lambda e: self.update_pinn_lamp_specs_display())

        self.pinn_lamp_specs_label = ttk.Label(lamp_config_frame, text="", font=('Arial', 9))
        self.pinn_lamp_specs_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=5)

        ttk.Label(lamp_config_frame, text="Orientation:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=5)
        self.pinn_orientation = tk.StringVar(value='perpendicular')
        ttk.Radiobutton(lamp_config_frame, text="Perpendicular (X-axis)",
                       variable=self.pinn_orientation, value='perpendicular').grid(row=2, column=1, sticky='w')
        ttk.Radiobutton(lamp_config_frame, text="Parallel (Y-axis)",
                       variable=self.pinn_orientation, value='parallel').grid(row=3, column=1, sticky='w')

        # 2. Data Status
        data_status_frame = ttk.LabelFrame(self.left_panel_pinn, text="Data Status", padding=10)
        data_status_frame.pack(fill='x', pady=5)

        self.pinn_data_status_label = ttk.Label(data_status_frame, text="No data loaded",
                                               font=('Arial', 9), foreground='orange')
        self.pinn_data_status_label.pack(anchor='w')

        # 3. Height Selection
        height_selection_frame = ttk.LabelFrame(self.left_panel_pinn, text="Height Selection", padding=10)
        height_selection_frame.pack(fill='x', pady=5)

        ttk.Label(height_selection_frame, text="Training Heights:",
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky='w', pady=5)

        for i in range(3):
            ttk.Label(height_selection_frame, text=f"Height {i+1}:").grid(row=i+1, column=0, sticky='w', padx=5)
            setattr(self, f'pinn_train_height{i+1}', tk.StringVar(value='None'))
            combo = ttk.Combobox(height_selection_frame,
                               textvariable=getattr(self, f'pinn_train_height{i+1}'),
                               values=['None'], state='readonly', width=18)
            combo.grid(row=i+1, column=1, padx=5, pady=2)
            setattr(self, f'pinn_train_height{i+1}_combo', combo)

        ttk.Label(height_selection_frame, text="Testing Height:",
                 font=('Arial', 10, 'bold')).grid(row=4, column=0, columnspan=2, sticky='w', pady=(10,5))
        ttk.Label(height_selection_frame, text="Height:").grid(row=5, column=0, sticky='w', padx=5)
        self.pinn_test_height = tk.StringVar(value='None')
        self.pinn_test_height_combo = ttk.Combobox(height_selection_frame,
                                                  textvariable=self.pinn_test_height,
                                                  values=['None'], state='readonly', width=18)
        self.pinn_test_height_combo.grid(row=5, column=1, padx=5, pady=2)

        self.pinn_height_note_label = ttk.Label(height_selection_frame, text="",
                                              font=('Arial', 8, 'italic'))
        self.pinn_height_note_label.grid(row=6, column=0, columnspan=2, sticky='w', pady=5)

        # 4. PINN Architecture
        pinn_arch_frame = ttk.LabelFrame(self.left_panel_pinn, text="PINN Architecture", padding=10)
        pinn_arch_frame.pack(fill='x', pady=5)

        ttk.Label(pinn_arch_frame, text="Number of Hidden Layers:",
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=5)
        self.pinn_num_layers = tk.IntVar(value=3)
        ttk.Spinbox(pinn_arch_frame, from_=1, to=5, textvariable=self.pinn_num_layers,
                   width=10, command=self.update_pinn_layer_inputs).grid(row=0, column=1, padx=5)

        # Container for layer neuron inputs
        self.pinn_layers_container = ttk.Frame(pinn_arch_frame)
        self.pinn_layers_container.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)

        ttk.Label(pinn_arch_frame, text="Activation Function:",
                 font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=5)
        self.pinn_activation = tk.StringVar(value='relu')
        ttk.Combobox(pinn_arch_frame, textvariable=self.pinn_activation,
                    values=['relu', 'tanh', 'sigmoid', 'elu'],
                    state='readonly', width=10).grid(row=2, column=1, padx=5)

        ttk.Label(pinn_arch_frame, text="Learning Rate:",
                 font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky='w', pady=5)
        self.pinn_learning_rate = tk.DoubleVar(value=0.001)
        ttk.Entry(pinn_arch_frame, textvariable=self.pinn_learning_rate,
                 width=12).grid(row=3, column=1, padx=5)

        ttk.Label(pinn_arch_frame, text="Epochs:",
                 font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky='w', pady=5)
        self.pinn_epochs = tk.IntVar(value=100)
        ttk.Spinbox(pinn_arch_frame, from_=10, to=1000, textvariable=self.pinn_epochs,
                   width=10, increment=10).grid(row=4, column=1, padx=5)

        ttk.Label(pinn_arch_frame, text="Batch Size:",
                 font=('Arial', 10, 'bold')).grid(row=5, column=0, sticky='w', pady=5)
        self.pinn_batch_size = tk.IntVar(value=16)
        ttk.Spinbox(pinn_arch_frame, from_=4, to=64, textvariable=self.pinn_batch_size,
                   width=10, increment=4).grid(row=5, column=1, padx=5)

        # Initialize layer inputs
        self.update_pinn_layer_inputs()

        # 5. Training Progress
        progress_frame = ttk.LabelFrame(self.left_panel_pinn, text="Training Progress", padding=10)
        progress_frame.pack(fill='x', pady=5)

        self.pinn_progress_var = tk.DoubleVar(value=0)
        self.pinn_progress_bar = ttk.Progressbar(progress_frame, variable=self.pinn_progress_var,
                                               maximum=100, length=200)
        self.pinn_progress_bar.pack(fill='x', pady=5)

        self.pinn_progress_label = ttk.Label(progress_frame, text="Ready",
                                           font=('Arial', 9), foreground='gray')
        self.pinn_progress_label.pack()

        # 6. Run Button
        button_frame = ttk.Frame(self.left_panel_pinn)
        button_frame.pack(fill='x', pady=10)

        self.pinn_run_button = ttk.Button(button_frame, text="Run PINN Estimation",
                                         command=self.run_pinn_estimation)
        self.pinn_run_button.pack(fill='x', pady=5)

        # Initialize displays
        self.update_pinn_lamp_specs_display()
        self.update_pinn_estimation_data_status()

    def update_pinn_layer_inputs(self):
        """Update layer neuron input fields based on number of layers"""
        # Clear existing inputs
        for widget in self.pinn_layers_container.winfo_children():
            widget.destroy()

        # Create new inputs
        num_layers = self.pinn_num_layers.get()
        self.pinn_layer_neurons = []

        default_neurons = [64, 32, 16, 8, 4]  # Default neuron counts

        for i in range(num_layers):
            ttk.Label(self.pinn_layers_container,
                     text=f"Layer {i+1} Neurons:").grid(row=i, column=0, sticky='w', pady=2)

            neuron_var = tk.IntVar(value=default_neurons[i] if i < len(default_neurons) else 16)
            self.pinn_layer_neurons.append(neuron_var)

            ttk.Spinbox(self.pinn_layers_container, from_=4, to=256,
                       textvariable=neuron_var, width=10, increment=4).grid(row=i, column=1, padx=5, pady=2)

    def update_pinn_lamp_specs_display(self):
        """Update lamp specifications display for PINN tab"""
        lamp_type = self.pinn_lamp_type.get()
        specs = self.led_specs[lamp_type]
        text = f"LEDs: {specs['num_leds']} | Length: {specs['length']}cm | Cost: {specs['cost']} TL"
        if lamp_type == 'Quantum':
            text += f" | Height offset: 13cm→0cm, 27/35cm→+4cm"
        self.pinn_lamp_specs_label.config(text=text)

        # Update data status
        self.update_pinn_estimation_data_status()

    def update_pinn_estimation_data_status(self):
        """Update data status in PINN estimation tab"""
        if not hasattr(self, 'pinn_data_status_label'):
            return

        if self.lamp_data and self.idx_points_df is not None:
            lamp_type = self.pinn_lamp_type.get()
            real_heights = [h + self.get_height_offset(lamp_type, h) for h in self.available_heights]

            text = f"Data loaded: {len(self.idx_points_df)} points, {len(self.available_heights)} heights\n"
            text += f"Nominal Heights: {self.available_heights}\n"

            if lamp_type == 'Quantum':
                text += f"Real Heights ({lamp_type}): {real_heights} (13cm→0, 27/35cm→+4cm)"
            else:
                text += f"Real Heights ({lamp_type}): {real_heights}"

            self.pinn_data_status_label.config(text=text, foreground='green')

            # Populate height selection comboboxes
            height_options = [f"{h}cm (Real: {h + self.get_height_offset(lamp_type, h)}cm)"
                            if lamp_type == 'Quantum' else f"{h}cm"
                            for h in self.available_heights]

            self.pinn_train_height1_combo['values'] = ['None'] + height_options
            self.pinn_train_height2_combo['values'] = ['None'] + height_options
            self.pinn_train_height3_combo['values'] = ['None'] + height_options
            self.pinn_test_height_combo['values'] = ['None'] + height_options

            # Set defaults
            if len(height_options) >= 2:
                self.pinn_train_height1.set(height_options[0])
                self.pinn_train_height2.set(height_options[1])
            if len(height_options) >= 3:
                self.pinn_test_height.set(height_options[2])

            # Update note
            if lamp_type == 'Quantum':
                self.pinn_height_note_label.config(
                    text=f"Note: For {lamp_type}, 13cm→real=13cm, 27cm→real=31cm, 35cm→real=39cm",
                    foreground='blue')
            else:
                self.pinn_height_note_label.config(text="")
        else:
            self.pinn_data_status_label.config(
                text="No data loaded. Please load data from Visualization tab first.",
                foreground='orange')

    def run_pinn_estimation(self):
        """Run PINN parameter estimation (Physics-Informed Neural Network)"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as e:
            messagebox.showerror("Error", f"Required library not found: {str(e)}\n\nPlease install: pip install tensorflow")
            return

        # Check if data is loaded
        if not self.lamp_data or self.idx_points_df is None:
            messagebox.showerror("Error", "Please load data from the Visualization tab first!")
            return

        # Get lamp configuration
        lamp_type = self.pinn_lamp_type.get()
        orientation = self.pinn_orientation.get()

        # Parse selected heights
        def parse_height(height_str):
            if height_str == 'None' or not height_str:
                return None
            return int(height_str.split('cm')[0])

        train_heights = []
        for i in range(1, 4):
            h = parse_height(getattr(self, f'pinn_train_height{i}').get())
            if h is not None:
                train_heights.append(h)

        test_height = parse_height(self.pinn_test_height.get())

        if len(train_heights) == 0:
            messagebox.showerror("Error", "Please select at least one Training Height!")
            return

        # Get PINN parameters
        num_layers = self.pinn_num_layers.get()
        layer_neurons = [var.get() for var in self.pinn_layer_neurons]
        activation = self.pinn_activation.get()
        learning_rate = self.pinn_learning_rate.get()
        epochs = self.pinn_epochs.get()
        batch_size = self.pinn_batch_size.get()

        # Show confirmation
        train_real_heights = [h + self.get_height_offset(lamp_type, h) for h in train_heights]
        info_msg = f"Lamp Type: {lamp_type}\n"
        info_msg += f"Orientation: {orientation}\n\n"
        info_msg += f"Training Heights: {train_heights} cm (Real: {train_real_heights} cm)\n"
        if test_height:
            test_real = test_height + self.get_height_offset(lamp_type, test_height)
            info_msg += f"Testing Height: {test_height} cm (Real: {test_real} cm)\n\n"
        info_msg += f"PINN Architecture: h → FFNN({' → '.join(map(str, layer_neurons))}) → [A]\n"
        info_msg += f"Physics: PPFD = A × Σ[h/R³ × exp(-α·R)]\n"
        info_msg += f"Activation: {activation}, LR: {learning_rate}, Epochs: {epochs}"

        messagebox.showinfo("PINN Estimation Setup", info_msg)

        # Prepare training data
        x_train_list = []
        y_train_list = []
        h_train_list = []
        ppfd_train_list = []
        train_idx_list = []

        for height in train_heights:
            lamp_df = self.lamp_data[height]
            merged_df = pd.merge(lamp_df, self.idx_points_df, on='idx', how='inner')

            x_coords = merged_df['X'].values
            y_coords = merged_df['Y'].values
            ppfd_values = merged_df[lamp_type].values * 0.8
            idx_values = merged_df['idx'].values
            real_height = height + self.get_height_offset(lamp_type, height)

            for x, y, ppfd, idx in zip(x_coords, y_coords, ppfd_values, idx_values):
                x_train_list.append(x)
                y_train_list.append(y)
                h_train_list.append(real_height)
                ppfd_train_list.append(ppfd)
                train_idx_list.append(idx)

        x_train = np.array(x_train_list)
        y_train = np.array(y_train_list)
        h_train = np.array(h_train_list)
        ppfd_train = np.array(ppfd_train_list)
        train_idx = np.array(train_idx_list)

        # Prepare testing data
        x_test = None
        y_test = None
        h_test = None
        ppfd_test = None
        test_idx = None
        if test_height:
            x_test_list = []
            y_test_list = []
            h_test_list = []
            ppfd_test_list = []
            test_idx_list = []

            test_lamp_df = self.lamp_data[test_height]
            test_merged = pd.merge(test_lamp_df, self.idx_points_df, on='idx', how='inner')

            x_coords = test_merged['X'].values
            y_coords = test_merged['Y'].values
            ppfd_values = test_merged[lamp_type].values * 0.8
            idx_values = test_merged['idx'].values
            test_real_height = test_height + self.get_height_offset(lamp_type, test_height)

            for x, y, ppfd, idx in zip(x_coords, y_coords, ppfd_values, idx_values):
                x_test_list.append(x)
                y_test_list.append(y)
                h_test_list.append(test_real_height)
                ppfd_test_list.append(ppfd)
                test_idx_list.append(idx)

            x_test = np.array(x_test_list)
            y_test = np.array(y_test_list)
            h_test = np.array(h_test_list)
            ppfd_test = np.array(ppfd_test_list)
            test_idx = np.array(test_idx_list)

        # Create LED positions based on lamp configuration
        lamp_specs = self.led_specs[lamp_type]
        num_leds = lamp_specs['num_leds']
        lamp_length = lamp_specs['length']

        positions = np.linspace(-lamp_length/2, lamp_length/2, num_leds)
        if orientation == 'perpendicular':
            x_leds = positions
            y_leds = np.zeros(num_leds)
        else:  # parallel
            x_leds = np.zeros(num_leds)
            y_leds = positions

        led_positions = np.column_stack([x_leds, y_leds])

        # Create PINN Model
        model = self._create_pinn_model(layer_neurons, activation)

        # Reset progress
        self.pinn_progress_var.set(0)
        self.pinn_progress_label.config(text="Training PINN...", foreground='blue')
        self.root.update()

        # Train PINN model
        history = self._train_pinn_model(
            model, x_train, y_train, h_train, ppfd_train, led_positions,
            epochs, batch_size, learning_rate
        )

        # Make predictions on training data
        y_train_pred = self._predict_pinn(model, x_train, y_train, h_train, led_positions)

        # Prepare results
        X_train = np.column_stack([x_train, y_train, h_train])
        results = {
            'model': model,
            'led_positions': led_positions,
            'lamp_type': lamp_type,
            'orientation': orientation,
            'train_heights': train_heights,
            'test_height': test_height,
            'num_layers': num_layers,
            'layer_neurons': layer_neurons,
            'activation': activation,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'X_train': X_train,
            'y_train': ppfd_train,
            'y_train_pred': y_train_pred,
            'train_x_coords': x_train,
            'train_y_coords': y_train,
            'train_lamp_heights': h_train,
            'train_idx': train_idx,
            'n_train': len(ppfd_train),
            'history': history
        }

        # Add test data if exists
        if test_height is not None:
            y_test_pred = self._predict_pinn(model, x_test, y_test, h_test, led_positions)
            X_test = np.column_stack([x_test, y_test, h_test])

            results['X_test'] = X_test
            results['y_test'] = ppfd_test
            results['y_test_pred'] = y_test_pred
            results['test_x_coords'] = x_test
            results['test_y_coords'] = y_test
            results['test_lamp_heights'] = h_test
            results['test_idx'] = test_idx
            results['n_test'] = len(ppfd_test)

        # Update progress
        self.pinn_progress_var.set(100)
        self.pinn_progress_label.config(text="Training completed!", foreground='green')

        # Display results
        self.display_pinn_results(results)

        messagebox.showinfo("Success", "PINN estimation completed successfully!")

    def _create_pinn_model(self, layer_neurons, activation):
        """Create Physics-Informed Neural Network Model"""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        class FlexiblePINNModel(keras.Model):
            def __init__(self, layer_sizes, activation_fn):
                super(FlexiblePINNModel, self).__init__()

                # FFNN: h → [A]
                ffnn_layers = []
                for i, size in enumerate(layer_sizes):
                    if i == 0:
                        ffnn_layers.append(layers.Dense(size, activation=activation_fn, input_shape=(1,)))
                    else:
                        ffnn_layers.append(layers.Dense(size, activation=activation_fn))
                ffnn_layers.append(layers.Dense(1, activation='linear'))

                self.ffnn = keras.Sequential(ffnn_layers, name='parameter_network')
                self.alpha_raw = tf.Variable(0.0, trainable=True, name='alpha_raw')

            def call(self, inputs, led_positions, training=False):
                x_points = inputs[:, 0:1]
                y_points = inputs[:, 1:2]
                h_points = inputs[:, 2:3]

                # FFNN: h → [A_raw]
                A_raw = self.ffnn(h_points, training=training)
                A = tf.nn.softplus(A_raw)

                # α global
                alpha = 0.01 + 0.5 * tf.nn.sigmoid(self.alpha_raw)
                alpha = tf.broadcast_to(alpha, tf.shape(A))

                # Physics: PPFD = A × Σ[h/R³ × exp(-α·R)]
                ppfd = self._physics_model(x_points, y_points, h_points, A, alpha, led_positions)
                return ppfd

            def _physics_model(self, x_points, y_points, h_points, A, alpha, led_positions):
                led_positions = tf.cast(led_positions, tf.float32)
                x_leds = led_positions[:, 0]
                y_leds = led_positions[:, 1]

                dx = x_points - x_leds
                dy = y_points - y_leds
                dz = -h_points

                R = tf.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)
                h_over_R3 = h_points / (R ** 3 + 1e-8)
                exp_term = tf.exp(-alpha * R)
                sum_contribution = tf.reduce_sum(h_over_R3 * exp_term, axis=1, keepdims=True)
                ppfd = A * sum_contribution
                return ppfd

        return FlexiblePINNModel(layer_neurons, activation)

    def _train_pinn_model(self, model, x_train, y_train, h_train, ppfd_train, led_positions,
                          epochs, batch_size, learning_rate):
        """Train PINN model with custom training loop"""
        import tensorflow as tf
        from tensorflow import keras

        X_train = np.column_stack([x_train, y_train, h_train]).astype(np.float32)
        y_train_reshaped = ppfd_train.reshape(-1, 1).astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_reshaped))
        dataset = dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = keras.losses.MeanSquaredError()

        history = {'loss': [], 'mae': []}

        for epoch in range(epochs):
            epoch_loss = []
            epoch_mae = []

            for batch_x, batch_y in dataset:
                with tf.GradientTape() as tape:
                    predictions = model(batch_x, led_positions=led_positions, training=True)
                    loss = loss_fn(batch_y, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                epoch_loss.append(loss.numpy())
                mae = tf.reduce_mean(tf.abs(batch_y - predictions)).numpy()
                epoch_mae.append(mae)

            avg_loss = np.mean(epoch_loss)
            avg_mae = np.mean(epoch_mae)

            history['loss'].append(avg_loss)
            history['mae'].append(avg_mae)

            # Update progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                progress = ((epoch + 1) / epochs) * 100
                self.pinn_progress_var.set(progress)
                self.pinn_progress_label.config(
                    text=f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}",
                    foreground='blue')
                self.root.update()

        return history

    def _predict_pinn(self, model, x, y, h, led_positions):
        """Make predictions using PINN model"""
        import tensorflow as tf

        X = np.column_stack([x, y, h]).astype(np.float32)
        ppfd_pred = model(X, led_positions=led_positions, training=False).numpy().flatten()
        return ppfd_pred

    def display_pinn_results(self, results):
        """Display PINN estimation results with detailed visualizations"""
        # Clear right panel
        for widget in self.right_panel_pinn.winfo_children():
            widget.destroy()

        # Create canvas with scrollbar (same as Linear Regression)
        canvas = tk.Canvas(self.right_panel_pinn, bg='white')
        scrollbar = ttk.Scrollbar(self.right_panel_pinn, orient="vertical", command=canvas.yview)

        # Create a frame inside the canvas
        scrollable_frame = ttk.Frame(canvas)

        # Configure the canvas
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Main container
        main_container = scrollable_frame

        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(title_frame, text="PINN Estimation Results",
                 font=('Arial', 14, 'bold')).pack()

        # Top section: Summary
        summary_frame = ttk.LabelFrame(main_container, text="Summary", padding=10)
        summary_frame.pack(fill='x', padx=10, pady=5)

        # Left and right columns
        left_summary = ttk.Frame(summary_frame)
        left_summary.pack(side='left', fill='both', expand=True, padx=10)

        right_summary = ttk.Frame(summary_frame)
        right_summary.pack(side='right', fill='both', expand=True, padx=10)

        # Left column: Model Architecture
        lamp_type = results['lamp_type']
        train_real_heights = [h + self.get_height_offset(lamp_type, h) for h in results['train_heights']]
        test_height = results['test_height']

        ttk.Label(left_summary, text=f"Lamp Type: {lamp_type}", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(left_summary, text=f"Orientation: {results['orientation'].upper()}", font=('Arial', 10)).pack(anchor='w', pady=2)

        ttk.Label(left_summary, text=f"Training Heights (nominal): {results['train_heights']} cm", font=('Arial', 9)).pack(anchor='w', pady=1)
        if lamp_type == 'Quantum':
            ttk.Label(left_summary, text=f"Training Heights (real): {train_real_heights} cm", font=('Arial', 9, 'bold'), foreground='blue').pack(anchor='w', pady=1)

        if test_height is not None:
            ttk.Label(left_summary, text=f"Testing Height (nominal): {test_height} cm", font=('Arial', 9)).pack(anchor='w', pady=1)
            if lamp_type == 'Quantum':
                test_real_height = test_height + self.get_height_offset(lamp_type, test_height)
                ttk.Label(left_summary, text=f"Testing Height (real): {test_real_height} cm", font=('Arial', 9, 'bold'), foreground='blue').pack(anchor='w', pady=1)

        ttk.Label(left_summary, text="", font=('Arial', 2)).pack(anchor='w', pady=1)

        # Model architecture
        layer_str = ' → '.join(map(str, results['layer_neurons']))
        ttk.Label(left_summary, text=f"PINN Architecture:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(left_summary, text=f"  h → FFNN({layer_str}) → A", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(left_summary, text=f"  Physics: PPFD = A × Σ[h/R³ × exp(-α·R)]", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(left_summary, text=f"  Activation: {results['activation']}", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(left_summary, text=f"  Learning Rate: {results['learning_rate']}", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(left_summary, text=f"  Epochs: {results['epochs']}, Batch Size: {results['batch_size']}", font=('Arial', 9)).pack(anchor='w')

        # Right column: Performance Metrics
        y_train = results['y_train']
        y_train_pred = results['y_train_pred']
        train_r2, train_rmse, train_mae = self.calculate_metrics(y_train, y_train_pred)

        ttk.Label(right_summary, text="Training Performance:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
        ttk.Label(right_summary, text=f"  R²: {train_r2:.6f}", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(right_summary, text=f"  RMSE: {train_rmse:.4f} µmol/m²/s", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(right_summary, text=f"  MAE: {train_mae:.4f} µmol/m²/s", font=('Arial', 9)).pack(anchor='w')
        ttk.Label(right_summary, text=f"  Samples: {results['n_train']}", font=('Arial', 9)).pack(anchor='w')

        if test_height is not None:
            y_test = results['y_test']
            y_test_pred = results['y_test_pred']
            test_r2, test_rmse, test_mae = self.calculate_metrics(y_test, y_test_pred)

            ttk.Label(right_summary, text="", font=('Arial', 2)).pack(anchor='w', pady=1)
            ttk.Label(right_summary, text="Testing Performance:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=2)
            ttk.Label(right_summary, text=f"  R²: {test_r2:.6f}", font=('Arial', 9)).pack(anchor='w')
            ttk.Label(right_summary, text=f"  RMSE: {test_rmse:.4f} µmol/m²/s", font=('Arial', 9)).pack(anchor='w')
            ttk.Label(right_summary, text=f"  MAE: {test_mae:.4f} µmol/m²/s", font=('Arial', 9)).pack(anchor='w')
            ttk.Label(right_summary, text=f"  Samples: {results['n_test']}", font=('Arial', 9)).pack(anchor='w')

        # Graphs section
        graph_frame = ttk.Frame(main_container)
        graph_frame.pack(fill='x', padx=10, pady=10)

        # Split training data by height and create plots
        self._create_pinn_plots(graph_frame, results)

        # Interactive Prediction section
        interactive_frame = ttk.LabelFrame(main_container, text="Interactive Prediction Calculator", padding=10)
        interactive_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(interactive_frame, text="Enter coordinates to predict PPFD:",
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=6, pady=5)

        ttk.Label(interactive_frame, text="X (cm):").grid(row=1, column=0, padx=5, pady=5)
        x_entry = ttk.Entry(interactive_frame, width=10)
        x_entry.grid(row=1, column=1, padx=5, pady=5)
        x_entry.insert(0, "0")

        ttk.Label(interactive_frame, text="Y (cm):").grid(row=1, column=2, padx=5, pady=5)
        y_entry = ttk.Entry(interactive_frame, width=10)
        y_entry.grid(row=1, column=3, padx=5, pady=5)
        y_entry.insert(0, "0")

        ttk.Label(interactive_frame, text="H (cm):").grid(row=1, column=4, padx=5, pady=5)
        h_entry = ttk.Entry(interactive_frame, width=10)
        h_entry.grid(row=1, column=5, padx=5, pady=5)
        first_train_height = results['train_heights'][0]
        default_h = first_train_height + self.get_height_offset(lamp_type, first_train_height)
        h_entry.insert(0, str(default_h))

        result_label = ttk.Label(interactive_frame, text="Predicted PPFD: --",
                                font=('Arial', 12, 'bold'), foreground='darkgreen',
                                background='lightyellow', relief='solid', padding=10)
        result_label.grid(row=2, column=0, columnspan=6, pady=10, sticky='ew')

        def predict_ppfd():
            try:
                import tensorflow as tf
                x = float(x_entry.get())
                y = float(y_entry.get())
                h = float(h_entry.get())

                # Predict using PINN model with LED positions
                X_new = np.array([[x, y, h]], dtype=np.float32)
                y_pred = results['model'](X_new, led_positions=results['led_positions'], training=False).numpy()[0][0]

                result_label.config(text=f"Predicted PPFD: {y_pred:.2f} µmol/m²/s")
            except Exception as e:
                result_label.config(text=f"Error: {str(e)}")

        ttk.Button(interactive_frame, text="Calculate", command=predict_ppfd).grid(row=3, column=0, columnspan=6, pady=5)

        note_text = "Note: Enter X, Y coordinates and lamp height (H) in cm. PINN uses physics equations with LED positions."
        ttk.Label(interactive_frame, text=note_text,
                 font=('Arial', 8, 'italic'), foreground='gray',
                 wraplength=600).grid(row=4, column=0, columnspan=6, pady=5)

        # Save Model Section
        save_model_frame = ttk.LabelFrame(main_container, text="Save PINN Model", padding=10)
        save_model_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(save_model_frame, text="Save trained PINN model for later use:",
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=5)

        save_button_frame = ttk.Frame(save_model_frame)
        save_button_frame.pack(fill='x', pady=5)

        def save_pinn_model():
            try:
                import json
                from datetime import datetime
                import os

                # Create saved_models directory if it doesn't exist
                models_dir = os.path.join(os.getcwd(), "saved_models")
                os.makedirs(models_dir, exist_ok=True)

                # Fixed filenames per lamp type (no timestamp)
                h5_filename = f"pinn_model_{lamp_type}.h5"
                metadata_filename = f"pinn_model_{lamp_type}_metadata.json"
                h5_path = os.path.join(models_dir, h5_filename)
                metadata_path = os.path.join(models_dir, metadata_filename)

                # Save model weights to H5
                results['model'].save_weights(h5_path)

                # Prepare metadata (everything except the model)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metadata = {
                    'led_positions': results['led_positions'].tolist(),  # Convert numpy to list for JSON
                    'lamp_type': results['lamp_type'],
                    'orientation': results['orientation'],
                    'train_heights': results['train_heights'],
                    'test_height': results['test_height'],
                    'num_layers': results['num_layers'],
                    'layer_neurons': results['layer_neurons'],
                    'activation': results['activation'],
                    'learning_rate': results['learning_rate'],
                    'epochs': results['epochs'],
                    'batch_size': results['batch_size'],
                    'train_r2': float(train_r2),
                    'train_rmse': float(train_rmse),
                    'test_r2': float(test_r2) if test_height is not None else None,
                    'test_rmse': float(test_rmse) if test_height is not None else None,
                    'led_specs': self.led_specs[lamp_type],
                    'saved_date': timestamp
                }

                # Save metadata to JSON
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                messagebox.showinfo("Success",
                    f"PINN model saved successfully!\n\n"
                    f"Lamp Type: {lamp_type}\n"
                    f"Model weights: {h5_path}\n"
                    f"Metadata: {metadata_path}\n\n"
                    f"Note: Existing model for this lamp type has been overwritten.")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model:\n{str(e)}")

        save_button = ttk.Button(save_button_frame, text="Save Model",
                                command=save_pinn_model, width=20)
        save_button.pack(side='left', padx=5)

        save_info = ttk.Label(save_model_frame,
                             text="Saved model includes: PINN weights, LED positions, scalers, and configuration",
                             font=('Arial', 8, 'italic'), foreground='gray')
        save_info.pack(anchor='w', pady=5)

        # Additional section: IDX vs Predicted/Measured line plots (same as Linear Regression)
        line_plot_frame = ttk.Frame(main_container)
        line_plot_frame.pack(fill='x', padx=10, pady=10)

        # Get lamp_type and train_heights from results
        lamp_type_for_plots = results['lamp_type']
        train_heights_for_plots = results['train_heights']

        # Split training data by height
        train_data_by_height = []
        unique_train_heights = np.unique(results['train_lamp_heights'])

        for h in unique_train_heights:
            mask = results['train_lamp_heights'] == h
            train_data_by_height.append({
                'x_coords': results['train_x_coords'][mask],
                'y_coords': results['train_y_coords'][mask],
                'measured': results['y_train'][mask],
                'predicted': results['y_train_pred'][mask],
                'idx': np.where(mask)[0]  # Get indices
            })

        self._create_idx_line_plots(line_plot_frame, results, train_data_by_height, train_heights_for_plots, lamp_type_for_plots)

        # Bottom section: Detailed Tables for each height (same as Linear Regression)
        table_frame = ttk.Frame(main_container)
        table_frame.pack(fill='x', padx=10, pady=10)

        self._create_pinn_detailed_tables(table_frame, results)

    def _create_pinn_plots(self, parent, results):
        """Create plots for PINN results (similar to ANN)"""
        lamp_type = results['lamp_type']
        train_heights = results['train_heights']
        test_height = results['test_height']

        # Split training data by height
        train_data_by_height = []
        unique_train_heights = np.unique(results['train_lamp_heights'])

        for h in unique_train_heights:
            mask = results['train_lamp_heights'] == h
            train_data_by_height.append({
                'x_coords': results['train_x_coords'][mask],
                'y_coords': results['train_y_coords'][mask],
                'measured': results['y_train'][mask],
                'predicted': results['y_train_pred'][mask]
            })

        # Prepare plot data
        all_plot_data = []
        for i, data in enumerate(train_data_by_height):
            # Find corresponding nominal height
            nominal_h = train_heights[i] if i < len(train_heights) else None
            all_plot_data.append({
                'data': data,
                'nominal_height': nominal_h,
                'title': f'Training {i+1}'
            })

        # Add test data if exists
        if test_height is not None:
            test_data = {
                'x_coords': results['test_x_coords'],
                'y_coords': results['test_y_coords'],
                'measured': results['y_test'],
                'predicted': results['y_test_pred']
            }
            all_plot_data.append({
                'data': test_data,
                'nominal_height': test_height,
                'title': 'Testing'
            })

        # Create plots (max 2 per row)
        num_total_plots = len(all_plot_data)
        plots_per_row = 2
        num_rows = (num_total_plots + plots_per_row - 1) // plots_per_row

        for row in range(num_rows):
            plots_in_this_row = min(plots_per_row, num_total_plots - row * plots_per_row)
            fig = Figure(figsize=(16, 6), dpi=90)

            for col in range(plots_in_this_row):
                plot_idx = row * plots_per_row + col
                plot_info = all_plot_data[plot_idx]

                ax = fig.add_subplot(1, plots_per_row, col + 1)
                self._plot_pinn_comparison(ax, plot_info['data'], plot_info['nominal_height'],
                                         lamp_type, plot_info['title'])

            fig.tight_layout()

            fig_canvas = FigureCanvasTkAgg(fig, master=parent)
            fig_canvas.draw()
            fig_canvas.get_tk_widget().pack(fill='x', expand=False, pady=5)

    def _plot_pinn_comparison(self, ax, data, nominal_height, lamp_type, data_type):
        """Plot measured vs predicted for PINN (grid style)"""
        x_coords = data['x_coords']
        y_coords = data['y_coords']
        measured = data['measured']
        predicted = data['predicted']

        # Get unique X and Y for grid lines
        unique_x = sorted(set(x_coords))
        unique_y = sorted(set(y_coords))

        # Draw grid lines
        for y in unique_y:
            ax.axhline(y=y, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)
        for x in unique_x:
            ax.axvline(x=x, color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Calculate errors (percentage)
        errors = predicted - measured
        abs_errors = np.abs(errors)
        # Calculate percentage error (avoid division by zero)
        error_percentage = np.where(measured != 0, (abs_errors / measured) * 100, 0)

        # Plot points
        scatter = ax.scatter(x_coords, y_coords, c=error_percentage, cmap='YlGn',
                           s=180, edgecolor='black', linewidth=1.5, zorder=5)

        # Add text labels (positioned above the points)
        for i in range(len(x_coords)):
            ax.text(x_coords[i], y_coords[i] + 1.5, f'M:{measured[i]:.0f}\nP:{predicted[i]:.0f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold', color='black',
                   zorder=6)

        # Title
        height_offset = self.get_height_offset(lamp_type, nominal_height)
        real_height = nominal_height + height_offset
        if lamp_type == 'Quantum':
            title = f'{data_type}: {nominal_height}cm (Real: {real_height}cm)'
        else:
            title = f'{data_type}: {nominal_height}cm'

        ax.set_xlabel('X (cm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (cm)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        # Colorbar
        plt.colorbar(scatter, ax=ax, label='Error %')

    def _create_pinn_detailed_tables(self, parent, results):
        """Create combined table for all PINN training data and separate table for testing"""

        # Get data
        lamp_type = results['lamp_type']
        train_heights = results['train_heights']
        test_height = results['test_height']

        # Split training data by height using actual idx values
        train_data_by_height = []
        unique_train_heights = np.unique(results['train_lamp_heights'])

        for h in unique_train_heights:
            mask = results['train_lamp_heights'] == h
            train_data_by_height.append({
                'x_coords': results['train_x_coords'][mask],
                'y_coords': results['train_y_coords'][mask],
                'measured': results['y_train'][mask],
                'predicted': results['y_train_pred'][mask],
                'idx': results['train_idx'][mask]  # Use actual idx from data
            })

        # Create combined training table
        self._create_combined_training_table(parent, train_data_by_height, train_heights, lamp_type)

        # Create table for testing height (only if test_height is not None)
        if test_height is not None:
            test_data = {
                'x_coords': results['test_x_coords'],
                'y_coords': results['test_y_coords'],
                'measured': results['y_test'],
                'predicted': results['y_test_pred'],
                'idx': results['test_idx']  # Use actual idx from data
            }
            self._create_single_height_table(parent, test_data,
                                             test_height, lamp_type, 'Testing')


    def create_custom_layout_interface(self):
        """Create Custom LED Layout interface"""
        # Main container with two panels
        main_paned = ttk.PanedWindow(self.custom_layout_frame, orient='horizontal')
        main_paned.pack(fill='both', expand=True)

        # Left panel (controls)
        left_panel = ttk.Frame(main_paned, width=400)
        main_paned.add(left_panel, weight=0)

        # Right panel (results)
        self.right_panel_custom = ttk.Frame(main_paned)
        main_paned.add(self.right_panel_custom, weight=1)

        # === LEFT PANEL CONTENTS ===

        # Title
        ttk.Label(left_panel, text="Custom LED Layout Designer",
                 font=('Arial', 12, 'bold')).pack(pady=10)

        # LED Configuration
        config_frame = ttk.LabelFrame(left_panel, text="LED Configuration", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(config_frame, text="Number of LEDs:").grid(row=0, column=0, sticky='w', pady=5)
        self.custom_num_leds = tk.IntVar(value=3)
        ttk.Spinbox(config_frame, from_=1, to=10, textvariable=self.custom_num_leds,
                   width=10).grid(row=0, column=1, sticky='w', padx=5)

        ttk.Button(config_frame, text="Auto Distribute LEDs",
                  command=self.auto_distribute_leds).grid(row=1, column=0, columnspan=2,
                                                          pady=10, sticky='ew')

        # LED List (Editable Table)
        led_list_frame = ttk.LabelFrame(left_panel, text="LED List (Editable)", padding=10)
        led_list_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Create Treeview
        columns = ('LED#', 'Type', 'X(cm)', 'Y(cm)', 'Length(cm)', 'Cost(TL)')
        self.custom_led_tree = ttk.Treeview(led_list_frame, columns=columns,
                                            show='headings', height=10)

        for col in columns:
            self.custom_led_tree.heading(col, text=col)
            if col == 'LED#':
                self.custom_led_tree.column(col, width=50, anchor='center')
            elif col in ['Type']:
                self.custom_led_tree.column(col, width=80, anchor='center')
            elif col in ['X(cm)', 'Y(cm)']:
                self.custom_led_tree.column(col, width=60, anchor='center')
            else:
                self.custom_led_tree.column(col, width=80, anchor='center')

        scrollbar = ttk.Scrollbar(led_list_frame, orient='vertical',
                                 command=self.custom_led_tree.yview)
        self.custom_led_tree.configure(yscrollcommand=scrollbar.set)

        self.custom_led_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Bind double-click for editing
        self.custom_led_tree.bind('<Double-1>', self.edit_led_entry)

        # Edit instruction
        ttk.Label(led_list_frame, text="Double-click to edit Type/X/Y",
                 font=('Arial', 8, 'italic'), foreground='gray').pack(pady=5)

        # Global Settings
        settings_frame = ttk.LabelFrame(left_panel, text="Global Settings", padding=10)
        settings_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(settings_frame, text="Orientation:").grid(row=0, column=0, sticky='w', pady=5)
        self.custom_orientation = tk.StringVar(value='parallel')
        ttk.Radiobutton(settings_frame, text="Parallel to X axis", variable=self.custom_orientation,
                       value='parallel').grid(row=0, column=1, sticky='w')
        ttk.Radiobutton(settings_frame, text="Perpendicular to X axis", variable=self.custom_orientation,
                       value='perpendicular').grid(row=1, column=1, sticky='w')

        ttk.Label(settings_frame, text="Height (cm):").grid(row=2, column=0, sticky='w', pady=5)
        self.custom_height = tk.StringVar(value='20')
        height_entry = ttk.Entry(settings_frame, textvariable=self.custom_height, width=10, state='normal')
        height_entry.grid(row=2, column=1, sticky='w', padx=5)
        height_entry.delete(0, tk.END)  # Clear default
        height_entry.insert(0, '20')  # Insert default
        ttk.Label(settings_frame, text="(between 15-40 cm)", font=('Arial', 8, 'italic'),
                 foreground='gray').grid(row=2, column=2, sticky='w')

        ttk.Label(settings_frame, text="Grid Resolution:", font=('Arial', 9, 'bold')).grid(
            row=3, column=0, sticky='w', pady=5)
        ttk.Label(settings_frame, text="10cm (11x11 grid)").grid(row=3, column=1, sticky='w')

        # Action Buttons
        action_frame = ttk.Frame(left_panel)
        action_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(action_frame, text="Calculate PPFD Map",
                  command=self.calculate_custom_ppfd).pack(fill='x', pady=5)
        ttk.Button(action_frame, text="Clear All",
                  command=self.clear_custom_layout).pack(fill='x', pady=5)

        # Storage for LED data
        self.custom_leds = []  # List of dicts: {'type': 'Quantum', 'x': 50, 'y': 50}

    def auto_distribute_leds(self):
        """Auto-distribute LEDs with margin"""
        num_leds = self.custom_num_leds.get()
        orientation = self.custom_orientation.get()

        # Clear existing LEDs
        self.custom_leds = []
        self.custom_led_tree.delete(*self.custom_led_tree.get_children())

        # Calculate positions with margin (~10cm from edges)
        margin = 10
        usable_length = 100 - 2 * margin  # 80cm

        if num_leds == 1:
            positions = [50]  # Center
        else:
            positions = np.linspace(margin, 100 - margin, num_leds)

        # Create LEDs based on orientation
        for i, pos in enumerate(positions):
            if orientation == 'parallel':
                # Parallel to X axis (horizontal bars): distribute along Y axis (X=50 center line)
                led = {'type': '54V', 'x': 50, 'y': pos}
            else:
                # Perpendicular to X axis (vertical bars): distribute along X axis (Y=50 center line)
                led = {'type': '54V', 'x': pos, 'y': 50}

            self.custom_leds.append(led)
            self._update_led_tree_entry(i)

    def _update_led_tree_entry(self, idx):
        """Update a single LED entry in the tree"""
        led = self.custom_leds[idx]
        led_type = led['type']
        specs = self.led_specs[led_type]

        # Check if item exists
        items = self.custom_led_tree.get_children()
        values = (idx + 1, led_type, f"{led['x']:.1f}", f"{led['y']:.1f}",
                 specs['length'], specs['cost'])

        if idx < len(items):
            # Update existing
            self.custom_led_tree.item(items[idx], values=values)
        else:
            # Insert new
            self.custom_led_tree.insert('', 'end', values=values)

    def edit_led_entry(self, event):
        """Handle double-click to edit LED entry inline"""
        item = self.custom_led_tree.selection()
        if not item:
            return

        item = item[0]
        column = self.custom_led_tree.identify_column(event.x)
        col_idx = int(column.replace('#', '')) - 1
        col_name = self.custom_led_tree['columns'][col_idx]

        # Only allow editing Type, X, Y
        if col_name not in ['Type', 'X(cm)', 'Y(cm)']:
            return

        # Get current value and position
        led_idx = self.custom_led_tree.index(item)
        current_value = self.custom_led_tree.item(item, 'values')[col_idx]

        # Get cell bounding box
        x, y, width, height = self.custom_led_tree.bbox(item, column)

        def save_inline_edit(new_value_var, editor_widget):
            """Save the edited value"""
            try:
                new_val = new_value_var.get()

                if col_name == 'Type':
                    self.custom_leds[led_idx]['type'] = new_val
                elif col_name == 'X(cm)':
                    x_val = float(new_val)
                    if 0 <= x_val <= 100:
                        self.custom_leds[led_idx]['x'] = x_val
                    else:
                        messagebox.showerror("Error", "X must be between 0-100 cm")
                        editor_widget.destroy()
                        return
                elif col_name == 'Y(cm)':
                    y_val = float(new_val)
                    if 0 <= y_val <= 100:
                        self.custom_leds[led_idx]['y'] = y_val
                    else:
                        messagebox.showerror("Error", "Y must be between 0-100 cm")
                        editor_widget.destroy()
                        return

                self._update_led_tree_entry(led_idx)
                editor_widget.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid value!")
                editor_widget.destroy()

        # Create inline editor
        if col_name == 'Type':
            # Combobox for Type
            new_value_var = tk.StringVar(value=current_value)
            editor = ttk.Combobox(self.custom_led_tree, textvariable=new_value_var,
                                values=['Quantum', '12V', '54V'], state='readonly')
            editor.place(x=x, y=y, width=width, height=height)
            editor.focus()
            editor.bind('<<ComboboxSelected>>', lambda e: save_inline_edit(new_value_var, editor))
            editor.bind('<FocusOut>', lambda e: editor.destroy())
        else:
            # Entry for X/Y
            new_value_var = tk.StringVar(value=current_value)
            editor = ttk.Entry(self.custom_led_tree, textvariable=new_value_var)
            editor.place(x=x, y=y, width=width, height=height)
            editor.focus()
            editor.select_range(0, tk.END)
            editor.bind('<Return>', lambda e: save_inline_edit(new_value_var, editor))
            editor.bind('<Escape>', lambda e: editor.destroy())
            editor.bind('<FocusOut>', lambda e: save_inline_edit(new_value_var, editor))

    def clear_custom_layout(self):
        """Clear all custom LED layout data"""
        self.custom_leds = []
        self.custom_led_tree.delete(*self.custom_led_tree.get_children())

        # Clear right panel
        for widget in self.right_panel_custom.winfo_children():
            widget.destroy()

    def calculate_custom_ppfd(self):
        """Calculate PPFD map for custom LED layout using saved PINN models"""
        if len(self.custom_leds) == 0:
            messagebox.showerror("Error", "Please add LEDs first (use Auto Distribute or manual entry)")
            return

        try:
            import tensorflow as tf
            from tensorflow import keras
            import json
        except ImportError as e:
            messagebox.showerror("Error", f"Required library not found: {str(e)}")
            return

        # Get settings
        try:
            height = float(self.custom_height.get())
            if not (15 <= height <= 40):
                messagebox.showerror("Error", "Height must be between 15-40 cm!")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid height value! Please enter a number.")
            return

        orientation = self.custom_orientation.get()

        # Load PINN models for each LED type used
        models_dir = os.path.join(os.getcwd(), "saved_models")
        loaded_models = {}

        unique_types = set(led['type'] for led in self.custom_leds)

        for led_type in unique_types:
            h5_path = os.path.join(models_dir, f"pinn_model_{led_type}.h5")
            metadata_path = os.path.join(models_dir, f"pinn_model_{led_type}_metadata.json")

            if not os.path.exists(h5_path) or not os.path.exists(metadata_path):
                messagebox.showerror("Error",
                    f"PINN model not found for {led_type}!\n\n"
                    f"Please train and save a PINN model for {led_type} first in the PINN Estimation tab.")
                return

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Recreate model architecture
            model = self._create_pinn_model(metadata['layer_neurons'], metadata['activation'])

            # Build model by calling with dummy input (required for subclassed models)
            led_positions_array = np.array(metadata['led_positions'], dtype=np.float32)
            dummy_input = np.array([[50.0, 50.0, 20.0]], dtype=np.float32)  # x, y, h
            _ = model(dummy_input, led_positions=led_positions_array, training=False)

            # Now load weights
            model.load_weights(h5_path)

            # Store model and LED positions
            loaded_models[led_type] = {
                'model': model,
                'led_positions': led_positions_array,
                'metadata': metadata
            }

        # Create 10cm grid (11x11 = 121 points)
        grid_points = np.arange(0, 101, 10)
        X_grid, Y_grid = np.meshgrid(grid_points, grid_points)
        ppfd_map = np.zeros_like(X_grid, dtype=float)

        # Calculate PPFD for each grid point
        for i, y_point in enumerate(grid_points):
            for j, x_point in enumerate(grid_points):
                total_ppfd = 0

                # Sum contributions from all LEDs
                for led in self.custom_leds:
                    led_type = led['type']
                    led_x = led['x']
                    led_y = led['y']

                    # Get model for this LED type
                    model_data = loaded_models[led_type]
                    model = model_data['model']
                    led_positions_original = model_data['led_positions']

                    # Transform LED positions based on orientation and LED location
                    # Original LED positions are centered at (0, 0)
                    # We need to translate and rotate them based on LED location and orientation

                    if orientation == 'parallel':
                        # Parallel to X axis: LED extends along X, centered at (led_x, led_y)
                        led_positions = led_positions_original.copy()
                        led_positions[:, 0] += led_x  # Translate X
                        led_positions[:, 1] += led_y  # Translate Y (all same)
                    else:
                        # Perpendicular to X axis: LED extends along Y
                        # Rotate 90 degrees: (x, y) -> (-y, x)
                        led_positions = np.column_stack([
                            -led_positions_original[:, 1] + led_x,
                            led_positions_original[:, 0] + led_y
                        ])

                    # No height offset for custom LED layout - use height directly
                    # Predict PPFD
                    X_input = np.array([[x_point, y_point, height]], dtype=np.float32)
                    ppfd_contrib = model(X_input, led_positions=led_positions, training=False).numpy()[0][0]

                    total_ppfd += ppfd_contrib

                ppfd_map[i, j] = total_ppfd

        # Display results
        self.display_custom_layout_results(X_grid, Y_grid, ppfd_map, loaded_models, height)

    def display_custom_layout_results(self, X_grid, Y_grid, ppfd_map, loaded_models, height):
        """Display custom LED layout results"""
        # Clear right panel
        for widget in self.right_panel_custom.winfo_children():
            widget.destroy()

        # Create canvas with scrollbar
        canvas = tk.Canvas(self.right_panel_custom, bg='white')
        scrollbar = ttk.Scrollbar(self.right_panel_custom, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        main_container = scrollable_frame

        # Title
        ttk.Label(main_container, text="Custom LED Layout Results",
                 font=('Arial', 14, 'bold')).pack(pady=10)

        # === 1. LED Layout Visualization (Top View) ===
        layout_frame = ttk.LabelFrame(main_container, text="LED Layout (Top View)", padding=10)
        layout_frame.pack(fill='both', padx=10, pady=5)

        fig_layout = Figure(figsize=(6, 6))
        ax_layout = fig_layout.add_subplot(111)

        # Draw 1m² area
        ax_layout.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], 'k-', linewidth=2)
        ax_layout.set_xlim(-10, 110)
        ax_layout.set_ylim(-10, 110)
        ax_layout.set_xlabel('X (cm)', fontsize=12)
        ax_layout.set_ylabel('Y (cm)', fontsize=12)
        ax_layout.set_title('LED Placement', fontsize=14, fontweight='bold')
        ax_layout.grid(True, alpha=0.3)
        ax_layout.set_aspect('equal')

        # Color map for LED types
        led_colors = {'Quantum': 'blue', '12V': 'green', '54V': 'red'}

        orientation = self.custom_orientation.get()

        for i, led in enumerate(self.custom_leds):
            led_type = led['type']
            led_x = led['x']
            led_y = led['y']
            color = led_colors[led_type]
            length = self.led_specs[led_type]['length']

            # Draw LED as rectangle
            if orientation == 'parallel':
                # Parallel to X: horizontal bar
                rect_x = led_x - length/2
                rect_y = led_y - 2
                rect_width = length
                rect_height = 4
            else:
                # Perpendicular: vertical bar
                rect_x = led_x - 2
                rect_y = led_y - length/2
                rect_width = 4
                rect_height = length

            rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                facecolor=color, edgecolor='black', linewidth=1.5,
                                alpha=0.7, label=led_type if i == 0 or led_type != self.custom_leds[i-1]['type'] else "")
            ax_layout.add_patch(rect)

            # Add LED number
            ax_layout.text(led_x, led_y, str(i+1), ha='center', va='center',
                          fontsize=10, fontweight='bold', color='white',
                          bbox=dict(boxstyle='circle', facecolor=color, edgecolor='black'))

        # Legend
        handles, labels = ax_layout.get_legend_handles_labels()
        if handles:
            ax_layout.legend(loc='upper right')

        fig_layout.tight_layout()

        canvas_layout = FigureCanvasTkAgg(fig_layout, layout_frame)
        canvas_layout.draw()
        canvas_layout.get_tk_widget().pack()

        # === 2. PPFD Heat Map ===
        heatmap_frame = ttk.LabelFrame(main_container, text="PPFD Heat Map (10cm Grid)", padding=10)
        heatmap_frame.pack(fill='both', padx=10, pady=5)

        fig_heat = Figure(figsize=(7, 6))
        ax_heat = fig_heat.add_subplot(111)

        # Create filled contour plot with better color scheme
        levels = np.linspace(ppfd_map.min(), ppfd_map.max(), 25)
        cp = ax_heat.contourf(X_grid, Y_grid, ppfd_map, levels=levels, cmap='YlOrRd', extend='both')
        cbar = fig_heat.colorbar(cp, ax=ax_heat)
        cbar.set_label('PPFD (µmol/m²/s)', fontsize=12)

        # Add contour lines with labels
        contour_lines = ax_heat.contour(X_grid, Y_grid, ppfd_map, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax_heat.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f')

        ax_heat.set_xlabel('X (cm)', fontsize=12)
        ax_heat.set_ylabel('Y (cm)', fontsize=12)
        ax_heat.set_title('PPFD Distribution', fontsize=14, fontweight='bold')
        ax_heat.set_aspect('equal')

        fig_heat.tight_layout()

        canvas_heat = FigureCanvasTkAgg(fig_heat, heatmap_frame)
        canvas_heat.draw()
        canvas_heat.get_tk_widget().pack()

        # === 3. Statistics Panel ===
        stats_frame = ttk.LabelFrame(main_container, text="Statistics", padding=10)
        stats_frame.pack(fill='x', padx=10, pady=5)

        # Calculate statistics
        total_cost = sum(self.led_specs[led['type']]['cost'] for led in self.custom_leds)
        avg_ppfd = np.mean(ppfd_map)
        min_ppfd = np.min(ppfd_map)
        max_ppfd = np.max(ppfd_map)

        # Uniformity metrics
        cv = (np.std(ppfd_map) / avg_ppfd) * 100 if avg_ppfd > 0 else 0
        uniformity_ratio = (min_ppfd / avg_ppfd) * 100 if avg_ppfd > 0 else 0

        orientation_text = "Parallel to X axis" if self.custom_orientation.get() == 'parallel' else "Perpendicular to X axis"

        stats_text = f"""
Total LEDs: {len(self.custom_leds)}
Total Cost: {total_cost} TL
Lamp Height: {height:.1f} cm
Orientation: {orientation_text}

PPFD Statistics:
  Average: {avg_ppfd:.2f} µmol/m²/s
  Minimum: {min_ppfd:.2f} µmol/m²/s
  Maximum: {max_ppfd:.2f} µmol/m²/s

Uniformity:
  CV (Coefficient of Variation): {cv:.2f}%
  Min/Avg Ratio: {uniformity_ratio:.2f}%
        """

        ttk.Label(stats_frame, text=stats_text, font=('Courier', 10),
                 justify='left').pack(anchor='w')

        # === 4. PPFD Data Table ===
        table_frame = ttk.LabelFrame(main_container, text="PPFD Data (121 Grid Points)", padding=10)
        table_frame.pack(fill='both', padx=10, pady=5)

        # Create Treeview
        columns = ('X(cm)', 'Y(cm)', 'PPFD(µmol/m²/s)')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='center')

        # Add data
        for i in range(11):
            for j in range(11):
                tree.insert('', 'end', values=(
                    f"{X_grid[i, j]:.0f}",
                    f"{Y_grid[i, j]:.0f}",
                    f"{ppfd_map[i, j]:.2f}"
                ))

        scrollbar_table = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar_table.set)

        tree.pack(side='left', fill='both', expand=True)
        scrollbar_table.pack(side='right', fill='y')


if __name__ == '__main__':
    root = tk.Tk()
    app = LEDVisualizationGUI(root)
    root.mainloop()
