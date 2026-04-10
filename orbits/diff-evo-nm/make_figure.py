"""Generate a multi-panel figure for the diff-evo-nm orbit."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import differential_evolution, minimize

# --- Global defaults ---
mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.2,
})

def evaluate_function(x, y):
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20

# Create mesh for contour plots
x = np.linspace(-5, 5, 300)
y = np.linspace(-5, 5, 300)
X, Y = np.meshgrid(x, y)
Z = evaluate_function(X, Y)

GLOBAL_MIN_X, GLOBAL_MIN_Y = -1.70408285, 0.67750781
GLOBAL_MIN_VAL = -1.5186858408

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

# --- Panel (a): Function landscape with global minimum ---
ax = axes[0]
cf = ax.contourf(X, Y, Z, levels=40, cmap='viridis')
ax.contour(X, Y, Z, levels=20, colors='white', linewidths=0.3, alpha=0.5)
ax.plot(GLOBAL_MIN_X, GLOBAL_MIN_Y, 'r*', markersize=18, markeredgecolor='white',
        markeredgewidth=1.5, label=f'Global min ({GLOBAL_MIN_X:.2f}, {GLOBAL_MIN_Y:.2f})')
fig.colorbar(cf, ax=ax, label='f(x, y)', shrink=0.9)
ax.set_title('(a) Function Landscape', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# --- Panel (b): Baseline vs DE+NM convergence points ---
ax = axes[1]
ax.contourf(X, Y, Z, levels=40, cmap='viridis', alpha=0.6)
ax.contour(X, Y, Z, levels=20, colors='white', linewidths=0.3, alpha=0.3)

# Run baseline random search 50 times
np.random.seed(42)
baseline_pts = []
for _ in range(50):
    best_x, best_y, best_val = 0.0, 0.0, float('inf')
    for __ in range(1000):
        xi = np.random.uniform(-5, 5)
        yi = np.random.uniform(-5, 5)
        vi = evaluate_function(xi, yi)
        if vi < best_val:
            best_x, best_y, best_val = xi, yi, vi
    baseline_pts.append((best_x, best_y))
baseline_pts = np.array(baseline_pts)

# Run DE+NM 50 times
obj = lambda xy: evaluate_function(xy[0], xy[1])
de_nm_pts = []
for _ in range(50):
    r = differential_evolution(obj, bounds=[(-5, 5), (-5, 5)],
                               strategy='best1bin', maxiter=100, popsize=15,
                               tol=1e-8, polish=False, init='latinhypercube')
    nm = minimize(obj, x0=r.x, method='Nelder-Mead',
                  options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000})
    de_nm_pts.append(nm.x if nm.fun <= r.fun else r.x)
de_nm_pts = np.array(de_nm_pts)

ax.scatter(baseline_pts[:, 0], baseline_pts[:, 1], c='orange', s=30, alpha=0.7,
           edgecolors='black', linewidths=0.5, label='Baseline (random)', zorder=3)
ax.scatter(de_nm_pts[:, 0], de_nm_pts[:, 1], c='red', s=30, alpha=0.7,
           edgecolors='white', linewidths=0.5, label='DE + NM', zorder=4)
ax.plot(GLOBAL_MIN_X, GLOBAL_MIN_Y, 'r*', markersize=18, markeredgecolor='white',
        markeredgewidth=1.5, zorder=5)
ax.set_title('(b) Convergence: Baseline vs DE+NM', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# --- Panel (c): Score comparison bar chart ---
ax = axes[2]
methods = ['Baseline\n(Random)', 'DE + NM\n(single)', 'DE + NM\n(10 restarts)']
scores = [1.058, 1.364, 1.4995]
colors = ['#888888', '#4dabf7', '#2b8a3e']
bars = ax.bar(methods, scores, color=colors, edgecolor='black', linewidth=0.8, width=0.6)

# Add value labels on bars
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.axhline(y=1.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Theoretical max (1.500)')
ax.set_ylim(0, 1.65)
ax.set_title('(c) Combined Score Comparison', fontweight='bold')
ax.set_ylabel('combined_score')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

fig.savefig('orbits/diff-evo-nm/figures/results.png', bbox_inches='tight')
plt.close(fig)
print("Figure saved to orbits/diff-evo-nm/figures/results.png")
