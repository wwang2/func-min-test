#!/bin/bash
# Reproduce the sa-anneal experiment from scratch
# Run from the repo root: bash orbits/sa-anneal/run.sh

set -e

echo "=== orbit/sa-anneal: Simulated Annealing ==="
echo ""
echo "--- Quick sanity check (10 runs) ---"
python3 orbits/sa-anneal/solution.py

echo ""
echo "--- Official evaluator (3 seeds, sequential) ---"
python3 -c "
import sys, numpy as np, time
sys.path.insert(0,'research/eval')
from evaluator import evaluate

scores = []
for seed in [42, 123, 7]:
    np.random.seed(seed)
    t0 = time.time()
    r = evaluate('orbits/sa-anneal/solution.py')
    elapsed = time.time() - t0
    s = r.metrics['combined_score']
    scores.append(s)
    print(f'Seed {seed}: combined_score={s:.5f}, time={elapsed:.1f}s')

print()
print(f'Mean: {np.mean(scores):.5f} +/- {np.std(scores):.5f}')
print(f'Baseline (random search): 1.43800')
"

echo ""
echo "--- Regenerate figures ---"
python3 -c "
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300,
})

def f(x, y):
    return np.sin(x)*np.cos(y) + np.sin(x*y) + (x**2+y**2)/20

xs = np.linspace(-5, 5, 300)
ys = np.linspace(-5, 5, 300)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)

def sa_trace(x0, y0, T_init=3.0, T_min=1e-4, alpha=0.96, step_init=1.2):
    x, y = x0, y0
    path_x, path_y = [x], [y]
    T = T_init
    cur = f(x, y)
    while T > T_min:
        step = step_init * np.sqrt(T/T_init)
        xn = np.clip(x + np.random.normal(0, step), -5, 5)
        yn = np.clip(y + np.random.normal(0, step), -5, 5)
        nv = f(xn, yn)
        d = nv - cur
        if d < 0 or np.random.random() < np.exp(-d/T):
            x, y, cur = xn, yn, nv
        if len(path_x) % 50 == 0:
            path_x.append(x); path_y.append(y)
        T *= alpha
    path_x.append(x); path_y.append(y)
    return path_x, path_y

np.random.seed(99)
path_x, path_y = sa_trace(2.0, -2.0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
ax = axes[0]
cf = ax.contourf(X, Y, Z, levels=30, cmap='RdYlBu_r')
ax.contour(X, Y, Z, levels=15, colors='k', alpha=0.2, linewidths=0.4)
plt.colorbar(cf, ax=ax, label='f(x,y)')
ax.set_title('(a) Function Landscape', fontweight='bold'); ax.set_xlabel('x'); ax.set_ylabel('y')
ax.plot(-1.704, 0.678, 'k*', ms=14, label='Global min (-1.704, 0.678)')
ax.legend(loc='upper right', frameon=True, fontsize=10)

ax = axes[1]
ax.contourf(X, Y, Z, levels=30, cmap='RdYlBu_r', alpha=0.7)
ax.contour(X, Y, Z, levels=15, colors='k', alpha=0.15, linewidths=0.4)
ax.plot(path_x, path_y, 'w-', lw=1, alpha=0.6)
ax.scatter(path_x, path_y, c=range(len(path_x)), cmap='plasma', s=15, zorder=3)
ax.plot(path_x[0], path_y[0], 'g^', ms=12, label=f'Start ({path_x[0]:.1f},{path_y[0]:.1f})')
ax.plot(path_x[-1], path_y[-1], 'r*', ms=14, label=f'End ({path_x[-1]:.3f},{path_y[-1]:.3f})')
ax.plot(-1.704, 0.678, 'k*', ms=14, label='Global min')
ax.set_title('(b) SA Trajectory (one chain)', fontweight='bold'); ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend(loc='upper right', frameon=True, fontsize=9)

ax = axes[2]
methods = ['Random Search\n(baseline)', 'SA v2\n(28 restarts)']
scores = [1.438, 1.4995]
bars = ax.bar(methods, scores, color=['#d62728', '#2ca02c'], edgecolor='k', linewidth=0.8, width=0.4)
ax.axhline(1.438, color='#d62728', linestyle='--', alpha=0.5, label='Baseline (1.438)')
ax.set_ylim(1.38, 1.55); ax.set_ylabel('combined_score')
ax.set_title('(c) Score Comparison', fontweight='bold')
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.003, f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.legend(frameon=False, fontsize=11)

fig.savefig('orbits/sa-anneal/figures/results.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
print('Figure saved to orbits/sa-anneal/figures/results.png')
"

echo ""
echo "=== Done. ==="
