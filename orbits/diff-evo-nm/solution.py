# EVOLVE-BLOCK-START
"""Function minimization using Differential Evolution + Nelder-Mead polish"""
import numpy as np
from scipy.optimize import differential_evolution, minimize


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Multi-restart Differential Evolution + Nelder-Mead polish.

    Each DE+NM run takes ~20ms, so within a 5-second budget we can afford
    many restarts. This drives the probability of missing the global minimum
    to near zero: if a single run has 86% chance of finding the global basin,
    then 10 independent restarts give 1 - 0.14^10 > 99.99% reliability.

    Args:
        iterations: Not used (kept for API compatibility)
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    search_bounds = [(bounds[0], bounds[1]), (bounds[0], bounds[1])]
    obj = lambda xy: evaluate_function(xy[0], xy[1])

    best_x, best_y, best_value = 0.0, 0.0, float('inf')

    # Run multiple independent DE restarts, keep the best
    num_restarts = 10
    for _ in range(num_restarts):
        # Differential Evolution: global search
        de_result = differential_evolution(
            obj,
            bounds=search_bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=None,
            polish=False,
            init='latinhypercube',
        )

        # Nelder-Mead: local polish for high precision
        nm_result = minimize(
            obj,
            x0=de_result.x,
            method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000},
        )

        candidate_value = min(de_result.fun, nm_result.fun)
        if candidate_value < best_value:
            if nm_result.fun <= de_result.fun:
                best_x, best_y = nm_result.x
                best_value = nm_result.fun
            else:
                best_x, best_y = de_result.x
                best_value = de_result.fun

    return best_x, best_y, best_value


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
