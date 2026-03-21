"""
Generate measurement data heatmaps for all LED types and heights.
Shows actual measured PPFD values overlaid on interpolated heatmap.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import pandas as pd
import os

os.makedirs('results', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

idx_pts = pd.read_csv('idx_to_points.csv')
idx_pts.columns = idx_pts.columns.str.strip().str.lower()

datasets = {}
for fname, h_nominal, h_real in [
    ('13_cm_led_height_ppfd_values.csv', 13, 13),
    ('27_cm_led_height_ppfd_values.csv', 27, 31),
    ('35_cm_led_height_ppfd_values.csv', 35, 39),
]:
    d = pd.read_csv(fname)
    d.columns = d.columns.str.strip().str.lower()
    df = pd.merge(d, idx_pts, on='idx', how='inner')
    datasets[(h_nominal, h_real)] = df

LED_TYPES = [
    ('quantum', 'Quantum LED', '#E65100'),
    ('54vgrow', '54V LED',     '#1565C0'),
    ('12vgrow', '12V LED',     '#2E7D32'),
]

HEIGHT_LABELS = [
    (13, 13, '13 cm'),
    (27, 31, '27 cm (gercek: 31 cm)'),
    (35, 39, '35 cm (gercek: 39 cm)'),
]

# ── Colour map: white → orange → deep red (like reference image) ──────────────

cmap = LinearSegmentedColormap.from_list(
    'ppfd_cmap',
    ['#FFF9C4', '#FFB300', '#E65100', '#BF360C'],
    N=256
)

# ── Interpolation grid ────────────────────────────────────────────────────────

XI = np.linspace(-50, 50, 300)
YI = np.linspace(-40, 40, 300)
XI, YI = np.meshgrid(XI, YI)


def make_panel(ax, x, y, values, title, led_color, h_nominal):
    """Draw one heatmap panel with measured values overlaid."""
    # Interpolate
    zi = griddata((x, y), values, (XI, YI), method='cubic')
    # Fill NaN edges with nearest
    zi_near = griddata((x, y), values, (XI, YI), method='nearest')
    zi = np.where(np.isnan(zi), zi_near, zi)

    vmin, vmax = values.min(), values.max()

    im = ax.imshow(
        zi,
        extent=[-50, 50, -40, 40],
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        alpha=0.92,
    )

    # Overlay measured value boxes
    box_w, box_h = 16, 10
    for xi, yi, vi in zip(x, y, values):
        norm_v = (vi - vmin) / (vmax - vmin + 1e-9)
        bg_col = cmap(norm_v)           # directly query colormap — reliable RGBA
        r, g, b, _ = bg_col
        luminance = 0.299*r + 0.587*g + 0.114*b
        txt_col = 'white' if luminance < 0.55 else '#2B1200'

        rect = patches.FancyBboxPatch(
            (xi - box_w/2, yi - box_h/2), box_w, box_h,
            boxstyle='round,pad=0.5',
            linewidth=1.2,
            edgecolor='white',
            facecolor=bg_col,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(xi, yi, f'{int(round(vi))}',
                ha='center', va='center',
                fontsize=7.5, fontweight='bold',
                color=txt_col, zorder=4)

    # LED strip indicator (horizontal bar at top)
    ax.axhspan(36, 40, color=led_color, alpha=0.25, zorder=1)
    ax.text(0, 38, 'LED', ha='center', va='center',
            fontsize=7, color=led_color, fontweight='bold', zorder=2)

    ax.set_xlim(-55, 55)
    ax.set_ylim(-44, 44)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    ax.set_xlabel('X (cm)', fontsize=8)
    ax.set_ylabel('Y (cm)', fontsize=8)
    ax.tick_params(labelsize=7)

    # Colorbar
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label('PPFD (umol/m2/s)', fontsize=7)

    # Stats annotation
    ax.text(0.02, 0.02,
            f'Ort: {values.mean():.0f}  Min: {values.min():.0f}  Max: {values.max():.0f}',
            transform=ax.transAxes,
            fontsize=6.5, color='black',
            bbox=dict(fc='white', alpha=0.7, ec='none', pad=2),
            zorder=5)


# ── Figure 1: Per LED type — all 3 heights in one row (3 figures) ─────────────

for led_col, led_name, led_color in LED_TYPES:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor('#F5F5F5')

    for ax, (h_nom, h_real, h_label) in zip(axes, HEIGHT_LABELS):
        df = datasets[(h_nom, h_real)]
        vals = df[led_col].values.astype(float) * 0.8
        make_panel(
            ax,
            df['x'].values, df['y'].values, vals,
            title=f'Yukseklik: {h_label}\n({led_name})',
            led_color=led_color,
            h_nominal=h_nom,
        )

    fig.suptitle(
        f'{led_name} — PPFD Olcum Dagilimi (Tum Yukseklikler)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    safe = led_col.replace('v', 'v')
    out = f'results/measurement_heatmap_{led_col}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {out}')


# ── Figure 2: All LED types at each height (3 figures) ───────────────────────

for h_nom, h_real, h_label in HEIGHT_LABELS:
    df = datasets[(h_nom, h_real)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor('#F5F5F5')

    for ax, (led_col, led_name, led_color) in zip(axes, LED_TYPES):
        vals = df[led_col].values.astype(float) * 0.8
        make_panel(
            ax,
            df['x'].values, df['y'].values, vals,
            title=f'{led_name}\nYukseklik: {h_label}',
            led_color=led_color,
            h_nominal=h_nom,
        )

    fig.suptitle(
        f'Tum LED Tipleri Karsilastirmasi — Yukseklik: {h_label}',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    out = f'results/measurement_heatmap_h{h_nom}cm.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {out}')


# ── Figure 3: Summary grid — 3 LED types × 3 heights (3×3) ──────────────────

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.patch.set_facecolor('#F5F5F5')

for row_i, (led_col, led_name, led_color) in enumerate(LED_TYPES):
    for col_i, (h_nom, h_real, h_label) in enumerate(HEIGHT_LABELS):
        ax = axes[row_i][col_i]
        df = datasets[(h_nom, h_real)]
        vals = df[led_col].values.astype(float) * 0.8
        make_panel(
            ax,
            df['x'].values, df['y'].values, vals,
            title=f'{led_name} | {h_label}',
            led_color=led_color,
            h_nominal=h_nom,
        )

fig.suptitle(
    'Tum LED Tipleri ve Yuksekliklerde PPFD Olcum Dagilimi\n(3 LED x 3 Yukseklik)',
    fontsize=14, fontweight='bold', y=1.01
)
plt.tight_layout()
out = 'results/measurement_heatmap_all.png'
plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Saved: {out}')

print('\nTum olcum isi haritalari kaydedildi: results/')
