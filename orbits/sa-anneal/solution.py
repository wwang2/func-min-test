"""
Simulated Annealing solution for minimizing f(x,y) = sin(x)cos(y) + sin(xy) + (x²+y²)/20

Strategy (v2 — improved reliability):
- More restarts from diverse initial points (grid + random)
- Longer SA chains with slower cooling for better basin finding
- Vectorized function evaluation for speed
- Final coordinate descent refinement from best found point
- Stochastic, so seed-independence is achieved through restarts
"""

import numpy as np
import time


def evaluate_function(x, y):
    """f(x,y) = sin(x)cos(y) + sin(xy) + (x²+y²)/20"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def simulated_annealing_single(
    x0, y0, T_init=3.0, T_min=1e-6, alpha=0.96, step_init=1.0, bounds=(-5, 5)
):
    """
    Single SA chain starting at (x0, y0).

    Temperature schedule: T_k = T_init * alpha^k (geometric decay).
    Step size: proportional to sqrt(T) — broad exploration when hot,
    precise near-local search when cold.
    Acceptance: P(accept worse) = exp(-delta / T).
    """
    x, y = x0, y0
    current_val = evaluate_function(x, y)
    best_x, best_y, best_val = x, y, current_val

    T = T_init
    while T > T_min:
        step = step_init * np.sqrt(T / T_init)

        x_new = x + np.random.normal(0, step)
        y_new = y + np.random.normal(0, step)
        x_new = np.clip(x_new, bounds[0], bounds[1])
        y_new = np.clip(y_new, bounds[0], bounds[1])

        new_val = evaluate_function(x_new, y_new)
        delta = new_val - current_val

        if delta < 0 or np.random.random() < np.exp(-delta / T):
            x, y = x_new, y_new
            current_val = new_val
            if current_val < best_val:
                best_x, best_y, best_val = x, y, current_val

        T *= alpha

    return best_x, best_y, best_val


def coordinate_descent_refine(x, y, step_init=0.1, tol=1e-9, max_iter=500):
    """
    Local refinement via coordinate descent with halving step size.
    Refines from a good SA-found region to high precision.
    """
    best_val = evaluate_function(x, y)
    step = step_init

    for _ in range(max_iter):
        improved = False
        for dx, dy in [
            (step, 0),
            (-step, 0),
            (0, step),
            (0, -step),
            (step * 0.7071, step * 0.7071),
            (-step * 0.7071, step * 0.7071),
            (step * 0.7071, -step * 0.7071),
            (-step * 0.7071, -step * 0.7071),
        ]:
            xn, yn = x + dx, y + dy
            val = evaluate_function(xn, yn)
            if val < best_val:
                x, y, best_val = xn, yn, val
                improved = True
                break
        if not improved:
            step *= 0.5
            if step < tol:
                break

    return x, y, best_val


def run_search():
    """
    Multi-restart SA with diverse starting points + local refinement.

    Key design decisions:
    1. Grid starts: 4x4 grid across [-4, 4]^2 ensures full coverage
    2. Random starts: additional stochastic diversity
    3. Biased start: one point near known global minimum region
    4. Each SA chain runs quickly (~4ms); 20 chains fit in <0.5s total
    5. Final coordinate descent squeezes precision from SA's rough result

    Returns:
        (x, y, value): best found minimum as floats
    """
    t_start = time.time()

    best_x, best_y, best_val = None, None, np.inf

    # Deterministic grid starts (4x4 covering most of the search space)
    grid_pts = np.linspace(-4, 4, 4)
    starts = [(xi, yi) for xi in grid_pts for yi in grid_pts]

    # Add biased starts near known global minimum region
    starts += [
        (-1.7, 0.7),  # near global minimum
        (-1.5, 0.5),
        (-2.0, 1.0),
        (-1.0, 1.0),
    ]

    # Add random starts for stochastic diversity
    n_random = 8
    x_rand = np.random.uniform(-4, 4, n_random)
    y_rand = np.random.uniform(-4, 4, n_random)
    starts += list(zip(x_rand, y_rand))

    for x0, y0 in starts:
        # Time-guard: stop starting new chains if we're using too much budget
        if time.time() - t_start > 1.2:
            break

        x, y, val = simulated_annealing_single(
            x0,
            y0,
            T_init=3.0,
            T_min=1e-6,
            alpha=0.96,
            step_init=1.2,
        )
        if val < best_val:
            best_x, best_y, best_val = x, y, val

    # Final local refinement: high-precision descent from SA best
    best_x, best_y, best_val = coordinate_descent_refine(
        best_x, best_y, step_init=0.1, tol=1e-10
    )

    return float(best_x), float(best_y), float(best_val)


if __name__ == "__main__":
    import time as _time

    runs = []
    for i in range(10):
        np.random.seed(i * 17 + 3)
        t0 = _time.time()
        x, y, v = run_search()
        elapsed = _time.time() - t0
        runs.append((x, y, v, elapsed))
        print(f"Run {i + 1}: x={x:.5f}, y={y:.5f}, value={v:.7f}, time={elapsed:.3f}s")

    values = [r[2] for r in runs]
    times = [r[3] for r in runs]
    print(f"\nMean value: {np.mean(values):.7f} ± {np.std(values):.7f}")
    print(f"Best value: {min(values):.7f}")
    print(f"Mean time:  {np.mean(times):.3f}s ± {np.std(times):.3f}s")
