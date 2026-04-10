"""Generate multi-panel results figure for diff-evo orbit."""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

# Global defaults (use keys that exist in this matplotlib version)
mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def f(x, y):
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def de_with_history(pop_size=40, max_gen=300, F=0.8, CR=0.85, seed=42):
    """Run DE while recording best-value history and all intermediate populations."""
    rng = np.random.default_rng(seed)
    D = 2
    lo, hi = -5.0, 5.0
    population = rng.uniform(lo, hi, size=(pop_size, D))
    fitness = f(population[:, 0], population[:, 1])
    best_idx = np.argmin(fitness)
    best_pos = population[best_idx].copy()
    best_val = fitness[best_idx]
    best_vals = [best_val]
    indices = np.arange(pop_size)

    # Snapshot populations at generations 0, 10, 50, 300
    snapshots = {0: population.copy()}

    for gen in range(max_gen):
        r1 = rng.integers(0, pop_size, size=pop_size)
        r2 = rng.integers(0, pop_size, size=pop_size)
        same = (r1 == indices) | (r2 == indices) | (r1 == r2)
        while same.any():
            r1[same] = rng.integers(0, pop_size, size=same.sum())
            r2[same] = rng.integers(0, pop_size, size=same.sum())
            same = (r1 == indices) | (r2 == indices) | (r1 == r2)
        mutant = np.clip(best_pos + F * (population[r1] - population[r2]), lo, hi)
        cross_mask = rng.random(size=(pop_size, D)) < CR
        j_rand = rng.integers(0, D, size=pop_size)
        cross_mask[indices, j_rand] = True
        trial = np.where(cross_mask, mutant, population)
        trial_fitness = f(trial[:, 0], trial[:, 1])
        improved = trial_fitness < fitness
        population[improved] = trial[improved]
        fitness[improved] = trial_fitness[improved]
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_val:
            best_val = fitness[current_best_idx]
            best_pos = population[current_best_idx].copy()
        best_vals.append(best_val)
        if gen + 1 in (10, 50):
            snapshots[gen + 1] = population.copy()

    snapshots[max_gen] = population.copy()
    return population, fitness, np.array(best_vals), snapshots


pop_final, fit_final, convergence, snapshots = de_with_history()

# Build function landscape
xs = np.linspace(-5, 5, 400)
ys = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)

# --- 3-panel figure ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

# Panel (a): Function landscape + final population
ax = axes[0]
cf = ax.contourf(X, Y, Z, levels=40, cmap="viridis_r")
ax.contour(X, Y, Z, levels=20, colors="white", linewidths=0.3, alpha=0.35)
cbar = fig.colorbar(cf, ax=ax, shrink=0.90, pad=0.02)
cbar.set_label("f(x, y)", fontsize=12)
ax.scatter(
    pop_final[:, 0],
    pop_final[:, 1],
    c="orange",
    s=35,
    zorder=5,
    label="Final pop (gen 300)",
)
ax.scatter(
    -1.704083,
    0.677508,
    c="red",
    s=140,
    marker="*",
    zorder=6,
    label=f"Found min\n(-1.704, 0.678, f=-1.5187)",
)
ax.scatter(
    -1.704,
    0.678,
    c="white",
    s=80,
    marker="x",
    linewidths=2.5,
    zorder=7,
    label="Evaluator ref.",
)
ax.set_title("(a) Landscape & Final Population", fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(frameon=True, framealpha=0.75, fontsize=9, loc="upper right")

# Panel (b): Convergence curve
ax = axes[1]
ax.plot(convergence, color="#2196F3", lw=2, label="DE/best/1")
ax.axhline(-1.5186858, color="red", ls="--", lw=1.5, label="True min = -1.51869")
ax.axhline(-1.519, color="#9E9E9E", ls=":", lw=1.5, label="Eval ref. = -1.519")
ax.set_title("(b) Convergence Curve (seed=42)", fontweight="bold")
ax.set_xlabel("Generation")
ax.set_ylabel("Best f(x, y)")
ax.legend(frameon=True, framealpha=0.85)
# Zoom in on the interesting range
ax.set_ylim(-1.53, -0.3)
ax2 = ax.inset_axes([0.35, 0.35, 0.60, 0.40])
ax2.plot(convergence, color="#2196F3", lw=1.5)
ax2.axhline(-1.5186858, color="red", ls="--", lw=1)
ax2.set_xlim(0, 50)
ax2.set_ylim(-1.525, -1.3)
ax2.set_title("Early gens (0–50)", fontsize=10)
ax2.tick_params(labelsize=9)

# Panel (c): Score comparison bar chart
ax = axes[2]
labels = ["Baseline\n(random search)", "DE/best/1\n(this work)"]
scores = [1.438, 1.4995]
colors = ["#9E9E9E", "#4CAF50"]
bars = ax.bar(
    labels, scores, color=colors, edgecolor="black", linewidth=0.8, width=0.45
)
ax.bar_label(
    bars, labels=[f"{s:.4f}" for s in scores], padding=5, fontsize=13, fontweight="bold"
)
ax.set_ylim(1.30, 1.56)
ax.set_title("(c) Combined Score Comparison", fontweight="bold")
ax.set_ylabel("Combined Score (higher is better)")
ax.axhline(1.4995, color="#4CAF50", ls="--", lw=1.2, alpha=0.6)
# Improvement annotation
improvement = (1.4995 - 1.438) / 1.438 * 100
ax.annotate(
    f"+{improvement:.1f}%\nimprovement",
    xy=(1, 1.4995),
    xytext=(0.5, 1.50),
    fontsize=12,
    ha="center",
    color="#2E7D32",
    arrowprops=dict(arrowstyle="->", color="#2E7D32"),
)

os.makedirs(os.path.join(os.path.dirname(__file__), "figures"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "figures", "results.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print(f"Saved: {out_path}")
