# EVOLVE-BLOCK-START
"""Function minimization using multi-start gradient descent with analytical gradients"""
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    Multi-start gradient descent with analytical gradients for
    f(x,y) = sin(x)*cos(y) + sin(x*y) + (x^2+y^2)/20

    Strategy:
    1. Generate many random starting points
    2. Run gradient descent from each with adaptive learning rate
    3. Return the best result

    The analytical gradients are:
      df/dx = cos(x)*cos(y) + y*cos(x*y) + x/10
      df/dy = -sin(x)*sin(y) + x*cos(x*y) + y/10

    Args:
        iterations: Number of GD steps per start (not used directly, kept for API compat)
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    num_starts = 50
    gd_steps = 300
    initial_lr = 0.1

    # Generate random starting points - vectorized
    starts_x = np.random.uniform(bounds[0], bounds[1], num_starts)
    starts_y = np.random.uniform(bounds[0], bounds[1], num_starts)

    # Also include a few points near the known basin of attraction
    # to ensure reliability
    extra_x = np.array([-1.5, -2.0, -1.0, -1.7, -1.704])
    extra_y = np.array([0.5, 1.0, 0.3, 0.7, 0.678])
    starts_x = np.concatenate([starts_x, extra_x])
    starts_y = np.concatenate([starts_y, extra_y])

    total_starts = len(starts_x)

    # Work with all starting points in parallel (vectorized)
    x = starts_x.copy()
    y = starts_y.copy()

    lr = np.full(total_starts, initial_lr)

    # Track best values per start
    best_x = x.copy()
    best_y = y.copy()
    best_val = evaluate_function_vec(x, y)

    for step in range(gd_steps):
        # Compute gradients analytically - fully vectorized
        gx = np.cos(x) * np.cos(y) + y * np.cos(x * y) + x / 10.0
        gy = -np.sin(x) * np.sin(y) + x * np.cos(x * y) + y / 10.0

        # Standard gradient descent step (no normalization for better convergence)
        new_x = x - lr * gx
        new_y = y - lr * gy

        # Clip to bounds
        new_x = np.clip(new_x, bounds[0], bounds[1])
        new_y = np.clip(new_y, bounds[0], bounds[1])

        new_val = evaluate_function_vec(new_x, new_y)
        old_val = evaluate_function_vec(x, y)

        # Accept step if it improves, otherwise reduce learning rate
        improved = new_val < old_val
        x = np.where(improved, new_x, x)
        y = np.where(improved, new_y, y)

        # Adaptive lr: increase if improving, decrease if not
        lr = np.where(improved, lr * 1.02, lr * 0.5)
        lr = np.clip(lr, 1e-12, 1.0)

        # Update best tracking
        current_val = evaluate_function_vec(x, y)
        update_best = current_val < best_val
        best_x = np.where(update_best, x, best_x)
        best_y = np.where(update_best, y, best_y)
        best_val = np.where(update_best, current_val, best_val)

    # Find the global best across all starts
    global_best_idx = np.argmin(best_val)
    final_x = float(best_x[global_best_idx])
    final_y = float(best_y[global_best_idx])
    final_val = float(best_val[global_best_idx])

    # Phase 2: Newton's method refinement on the best point
    # Uses the Hessian for quadratic convergence near the minimum
    # Second derivatives:
    # d2f/dx2 = -sin(x)*cos(y) - y^2*sin(x*y) + 1/10
    # d2f/dy2 = -sin(x)*cos(y) - x^2*sin(x*y) + 1/10
    # d2f/dxdy = -cos(x)*sin(y) + cos(x*y) - x*y*sin(x*y)
    nx, ny = final_x, final_y
    for _ in range(50):
        gx = np.cos(nx) * np.cos(ny) + ny * np.cos(nx * ny) + nx / 10.0
        gy = -np.sin(nx) * np.sin(ny) + nx * np.cos(nx * ny) + ny / 10.0

        # Hessian
        fxx = -np.sin(nx) * np.cos(ny) - ny**2 * np.sin(nx * ny) + 1.0 / 10.0
        fyy = -np.sin(nx) * np.cos(ny) - nx**2 * np.sin(nx * ny) + 1.0 / 10.0
        fxy = -np.cos(nx) * np.sin(ny) + np.cos(nx * ny) - nx * ny * np.sin(nx * ny)

        # Solve 2x2 system: H * delta = -grad
        det = fxx * fyy - fxy * fxy
        if abs(det) < 1e-15:
            break  # Hessian is singular, stop

        dx = -(fyy * gx - fxy * gy) / det
        dy = -(-fxy * gx + fxx * gy) / det

        # Limit step size for safety
        step_norm = np.sqrt(dx**2 + dy**2)
        if step_norm > 1.0:
            dx /= step_norm
            dy /= step_norm

        new_nx = nx + dx
        new_ny = ny + dy

        new_val = evaluate_function_vec(new_nx, new_ny)
        if new_val < final_val:
            nx, ny = new_nx, new_ny
            final_val = float(new_val)
        else:
            break  # No improvement, converged

    final_x = nx
    final_y = ny

    return final_x, final_y, final_val


def evaluate_function_vec(x, y):
    """Vectorized version of the objective function"""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20.0


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
