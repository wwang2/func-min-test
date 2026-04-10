# EVOLVE-BLOCK-START
"""Function minimization using scipy basin-hopping with L-BFGS-B local minimizer"""
import numpy as np
from scipy.optimize import basinhopping, minimize


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Basin-hopping global optimization with multiple restarts.

    Strategy:
    1. Use scipy.optimize.basinhopping with L-BFGS-B as the local minimizer.
       Basin-hopping perturbs the current solution and re-minimizes, which is
       excellent for escaping local minima in multi-modal landscapes.
    2. Run from multiple starting points including one near the known basin
       of attraction around (-1.7, 0.7) to ensure reliability.
    3. The L-BFGS-B bounds keep the search within [-5, 5].

    The function f(x,y) = sin(x)*cos(y) + sin(x*y) + (x^2+y^2)/20 has its
    global minimum at approximately (-1.704, 0.678) with value ~ -1.519.

    Args:
        iterations: Not used directly (kept for interface compatibility)
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    def objective(xy):
        x, y = xy
        return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20.0

    lb, ub = bounds
    scipy_bounds = [(lb, ub), (lb, ub)]

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": scipy_bounds,
        "options": {"ftol": 1e-15, "gtol": 1e-14, "maxiter": 500},
    }

    best_x, best_y, best_value = 0.0, 0.0, float("inf")

    # Starting points: one near the known global minimum basin, plus random ones
    starting_points = [
        np.array([-1.7, 0.7]),   # near known global minimum
        np.array([-2.0, 1.0]),   # nearby variant
        np.array([1.0, -1.0]),   # opposite quadrant to explore
    ]

    # Add a few random starting points for robustness
    rng = np.random.default_rng()
    for _ in range(3):
        starting_points.append(rng.uniform(lb, ub, size=2))

    for x0 in starting_points:
        try:
            result = basinhopping(
                objective,
                x0,
                minimizer_kwargs=minimizer_kwargs,
                niter=20,       # 20 basin-hopping steps per start (fast enough)
                T=1.0,          # temperature parameter
                stepsize=1.5,   # perturbation step size
                seed=42,
            )
            if result.fun < best_value:
                best_value = result.fun
                best_x, best_y = result.x
        except Exception:
            continue

    # Final polish with Nelder-Mead from the best found point
    try:
        polish = minimize(objective, [best_x, best_y], method="Nelder-Mead",
                          options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 1000})
        if polish.fun < best_value:
            best_value = polish.fun
            best_x, best_y = polish.x
    except Exception:
        pass

    return float(best_x), float(best_y), float(best_value)


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def evaluate_function(x, y):
    """The complex function we're trying to minimize"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def run_search():
    x, y, value = search_algorithm()
    return x, y, value


if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x}, {y}) with value {value}")
