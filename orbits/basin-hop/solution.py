"""
Basin Hopping algorithm for minimizing f(x,y) = sin(x)cos(y) + sin(xy) + (x²+y²)/20

Basin hopping works by alternating between:
1. Local minimization (Nelder-Mead simplex) to find the local basin minimum
2. Random perturbation to jump out of the current basin
3. Metropolis acceptance criterion to allow occasional uphill moves
4. Adaptive step size to maintain an effective acceptance rate

The key insight is that by replacing the rugged energy landscape with a
"staircase" of local minima (the basin-transformed surface), we turn a
multi-modal problem into one where global search is tractable.

Global minimum: approximately (-1.704, 0.678) with value ~ -1.519
"""

import numpy as np
import time


# --- Objective function ---
def f(x, y):
    """f(x,y) = sin(x)cos(y) + sin(xy) + (x²+y²)/20"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20.0


# --- Nelder-Mead simplex local optimizer (pure numpy) ---
def nelder_mead(func, x0, y0, max_iter=400, tol=1e-9, step=0.5):
    """
    Nelder-Mead simplex for 2D minimization.

    A simplex is a triangle (3 vertices) in 2D. Each iteration:
    - Sorts vertices by function value (best to worst)
    - Reflects the worst vertex through the centroid of the rest
    - Expands (if reflection was good) or contracts (if reflection was bad)
    - Shrinks the entire simplex toward the best vertex as a last resort
    """
    p0 = np.array([x0, y0])
    p1 = p0 + np.array([step, 0.0])
    p2 = p0 + np.array([0.0, step])
    simplex = np.array([p0, p1, p2])
    fvals = np.array([func(p[0], p[1]) for p in simplex])

    alpha = 1.0  # reflection coefficient
    gamma = 2.0  # expansion coefficient
    rho = 0.5  # contraction coefficient
    sigma = 0.5  # shrink coefficient

    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        # Convergence: function values nearly identical
        if np.max(np.abs(fvals - fvals[0])) < tol:
            break

        # Centroid of all vertices except worst
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr[0], xr[1])

        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < fvals[0]:
            # Expansion — try to go even further in this direction
            xe = centroid + gamma * (xr - centroid)
            fe = func(xe[0], xe[1])
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        else:
            # Contraction — pull back toward centroid
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = func(xc[0], xc[1])
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink entire simplex toward best vertex
                simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
                fvals[1:] = np.array([func(p[0], p[1]) for p in simplex[1:]])

    return simplex[0, 0], simplex[0, 1], fvals[0]


# --- Basin Hopping global optimizer ---
def basin_hopping(
    seed=None,
    n_hops=200,
    step_size=1.2,
    temperature=0.5,
    bounds=(-5.0, 5.0),
    n_starts=8,
    step_adapt_interval=10,
    target_accept=0.25,
    time_limit=4.0,
):
    """
    Basin Hopping for 2D global minimization.

    The algorithm transforms f(x) into a "basin-flattened" surface where
    each point is mapped to the value of the local minimum in its basin.
    This staircase surface can be explored with a random walk + Metropolis
    acceptance criterion, enabling escape from local traps.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    n_hops : int
        Maximum hops per restart.
    step_size : float
        Initial Gaussian step size for perturbations.
    temperature : float
        Metropolis temperature — controls willingness to accept worse basins.
    bounds : (float, float)
        Search domain for both x and y.
    n_starts : int
        Number of independent random restarts (best result is returned).
    step_adapt_interval : int
        How often to adapt step size based on acceptance rate.
    target_accept : float
        Target acceptance rate for step adaptation (~0.25 is standard).
    time_limit : float
        Wall-clock time limit in seconds (hard budget).
    """
    rng = np.random.RandomState(seed)
    t_start = time.time()

    global_best_x = 0.0
    global_best_y = 0.0
    global_best_val = np.inf

    for start_idx in range(n_starts):
        # Check time budget — leave a small margin for final polish
        if time.time() - t_start > time_limit * 0.85:
            break

        # Random starting position
        x0 = rng.uniform(bounds[0], bounds[1])
        y0 = rng.uniform(bounds[0], bounds[1])

        # Initial local minimization
        cx, cy, cv = nelder_mead(f, x0, y0, max_iter=400, step=0.8)

        best_x, best_y, best_val = cx, cy, cv
        current_x, current_y, current_val = cx, cy, cv

        accept_count = 0
        step = step_size
        hops_per_start = n_hops // n_starts

        for hop in range(hops_per_start):
            if time.time() - t_start > time_limit * 0.9:
                break

            # Gaussian perturbation in both dimensions
            dx = rng.randn() * step
            dy = rng.randn() * step
            nx = float(np.clip(current_x + dx, bounds[0], bounds[1]))
            ny = float(np.clip(current_y + dy, bounds[0], bounds[1]))

            # Descend to local minimum from perturbed point
            nx, ny, nv = nelder_mead(f, nx, ny, max_iter=400, step=0.4)

            # Metropolis acceptance: always accept improvements,
            # accept worsening moves with probability exp(-ΔE/T)
            delta = nv - current_val
            if delta < 0.0 or rng.random() < np.exp(-delta / temperature):
                current_x, current_y, current_val = nx, ny, nv
                accept_count += 1
                if nv < best_val:
                    best_x, best_y, best_val = nx, ny, nv

            # Adaptive step size: scale to maintain target acceptance rate
            if (hop + 1) % step_adapt_interval == 0:
                accept_rate = accept_count / step_adapt_interval
                if accept_rate > target_accept:
                    step = min(step * 1.2, 4.0)  # increase — too easy
                else:
                    step = max(step * 0.8, 0.05)  # decrease — too hard
                accept_count = 0

        if best_val < global_best_val:
            global_best_x = best_x
            global_best_y = best_y
            global_best_val = best_val

    # Final high-precision polish on best solution found
    gx, gy, gv = nelder_mead(
        f, global_best_x, global_best_y, max_iter=1000, tol=1e-12, step=0.1
    )
    if gv < global_best_val:
        global_best_x, global_best_y, global_best_val = gx, gy, gv

    return global_best_x, global_best_y, global_best_val


def run_search():
    """
    Entry point called by the evaluator. Returns (x, y, value).
    Uses a fresh random seed each call to maintain diversity across the
    10 evaluator trials.
    """
    seed = np.random.randint(0, 2**31 - 1)
    x, y, val = basin_hopping(
        seed=seed,
        n_hops=200,
        step_size=1.2,
        temperature=0.5,
        bounds=(-5.0, 5.0),
        n_starts=8,
        time_limit=4.0,
    )
    return float(x), float(y), float(val)


if __name__ == "__main__":
    print("=== Basin Hopping Sanity Test ===")
    print(f"Global min: x=-1.704, y=0.678, value=-1.519\n")

    times = []
    for i in range(10):
        t0 = time.time()
        x, y, v = run_search()
        elapsed = time.time() - t0
        times.append(elapsed)
        dist = np.sqrt((x + 1.704) ** 2 + (y - 0.678) ** 2)
        print(
            f"Run {i:2d}: x={x:+.5f}  y={y:+.5f}  v={v:.6f}  dist={dist:.5f}  t={elapsed:.3f}s"
        )

    import numpy as np

    print(
        f"\nTime: {np.mean(times):.3f}s ± {np.std(times):.3f}s (max={np.max(times):.3f}s)"
    )
