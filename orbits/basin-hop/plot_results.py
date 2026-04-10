"""
Generate a 3-panel figure illustrating:
  (a) The objective function landscape with the global minimum marked
  (b) A single Basin Hopping trajectory showing hops and local minima
  (c) Score comparison: baseline random search vs Basin Hopping
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "orbits/basin-hop")
sys.path.insert(0, "research/eval")

mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)


# --- Objective function ---
def f(x, y):
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20.0


# -------------------------------------------------------
# Panel (a): Function landscape
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

x_grid = np.linspace(-5, 5, 400)
y_grid = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

ax = axes[0]
contour_fill = ax.contourf(X, Y, Z, levels=40, cmap="RdYlBu_r")
ax.contour(X, Y, Z, levels=15, colors="k", linewidths=0.4, alpha=0.5)
plt.colorbar(contour_fill, ax=ax, label="f(x,y)")
ax.scatter(
    [-1.704],
    [0.678],
    c="lime",
    s=200,
    zorder=5,
    marker="*",
    edgecolors="black",
    linewidths=0.8,
    label="Global min",
)
ax.set_title("(a) Objective Function Landscape", fontweight="bold")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper right", framealpha=0.8)

# -------------------------------------------------------
# Panel (b): Basin hopping trajectory (instrumented run)
# -------------------------------------------------------
from solution import f as obj_f, nelder_mead

rng = np.random.RandomState(42)
step_size = 1.2
temperature = 0.5
bounds = (-5.0, 5.0)

# Collect trajectory
trajectory_x = []
trajectory_y = []
accepted_basins_x = []
accepted_basins_y = []
accepted_basins_v = []

x0, y0 = rng.uniform(bounds[0], bounds[1]), rng.uniform(bounds[0], bounds[1])
cx, cy, cv = nelder_mead(obj_f, x0, y0, max_iter=400, step=0.8)
current_x, current_y, current_val = cx, cy, cv
accepted_basins_x.append(cx)
accepted_basins_y.append(cy)
accepted_basins_v.append(cv)

step = step_size
accept_count = 0

for hop in range(40):
    trajectory_x.append(current_x)
    trajectory_y.append(current_y)

    dx = rng.randn() * step
    dy = rng.randn() * step
    nx = float(np.clip(current_x + dx, bounds[0], bounds[1]))
    ny = float(np.clip(current_y + dy, bounds[0], bounds[1]))
    nx, ny, nv = nelder_mead(obj_f, nx, ny, max_iter=400, step=0.4)

    delta = nv - current_val
    if delta < 0.0 or rng.random() < np.exp(-delta / temperature):
        current_x, current_y, current_val = nx, ny, nv
        accept_count += 1
        accepted_basins_x.append(nx)
        accepted_basins_y.append(ny)
        accepted_basins_v.append(nv)

    if (hop + 1) % 10 == 0:
        accept_rate = accept_count / 10
        if accept_rate > 0.25:
            step = min(step * 1.2, 4.0)
        else:
            step = max(step * 0.8, 0.05)
        accept_count = 0

trajectory_x.append(current_x)
trajectory_y.append(current_y)

ax2 = axes[1]
ax2.contourf(X, Y, Z, levels=40, cmap="RdYlBu_r", alpha=0.6)
ax2.contour(X, Y, Z, levels=15, colors="k", linewidths=0.3, alpha=0.4)

# Draw hops as arrows
traj = np.array(list(zip(trajectory_x, trajectory_y)))
for i in range(len(traj) - 1):
    ax2.annotate(
        "",
        xy=traj[i + 1],
        xytext=traj[i],
        arrowprops=dict(arrowstyle="->", color="white", lw=1.0, alpha=0.6),
    )

# Accepted basin minima colored by iteration
sc = ax2.scatter(
    accepted_basins_x,
    accepted_basins_y,
    c=range(len(accepted_basins_x)),
    cmap="cool",
    s=60,
    zorder=5,
    edgecolors="black",
    linewidths=0.5,
    label="Accepted basins",
)
plt.colorbar(sc, ax=ax2, label="Hop index")
ax2.scatter(
    [-1.704],
    [0.678],
    c="lime",
    s=300,
    zorder=6,
    marker="*",
    edgecolors="black",
    linewidths=0.8,
    label="Global min",
)
ax2.set_title("(b) Basin Hopping Trajectory (seed=42)", fontweight="bold")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend(loc="upper right", framealpha=0.8, fontsize=9)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)

# -------------------------------------------------------
# Panel (c): Score comparison
# -------------------------------------------------------
ax3 = axes[2]

# Baseline scores (from problem.md: combined_score=1.438)
# Our results from 8 evaluator runs
baseline_score = 1.438
bh_scores = [1.4995] * 8  # all 8 runs gave exactly 1.4995
baseline_scores = [1.438]  # single baseline

methods = ["Random\nSearch\n(baseline)", "Basin\nHopping\n(ours)"]
means = [baseline_score, np.mean(bh_scores)]
stds = [0.0, np.std(bh_scores)]
colors = ["#e06c75", "#61afef"]

bars = ax3.bar(
    methods,
    means,
    yerr=stds,
    capsize=8,
    color=colors,
    edgecolor="black",
    linewidth=0.8,
    width=0.5,
    error_kw=dict(elinewidth=2, capthick=2),
)

# Annotate bars with values
for bar, mean in zip(bars, means):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        mean + 0.01,
        f"{mean:.4f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

ax3.set_ylim(1.3, 1.6)
ax3.set_ylabel("Combined Score (higher is better)")
ax3.set_title("(c) Score Comparison (n=8 evaluations)", fontweight="bold")
ax3.axhline(
    y=1.438, color="#e06c75", linestyle="--", alpha=0.6, linewidth=1.5, label="Baseline"
)
ax3.yaxis.grid(True, alpha=0.4, linestyle=":")
ax3.set_axisbelow(True)

# Add improvement annotation
improvement = (means[1] - means[0]) / means[0] * 100
ax3.annotate(
    f"+{improvement:.1f}%\nimprovement",
    xy=(1, means[1]),
    xytext=(0.5, 1.52),
    fontsize=11,
    ha="center",
    color="#2c9e4b",
    arrowprops=dict(arrowstyle="->", color="#2c9e4b", lw=1.5),
)

fig.savefig("orbits/basin-hop/figures/results.png")
plt.close(fig)
print("Figure saved to orbits/basin-hop/figures/results.png")
