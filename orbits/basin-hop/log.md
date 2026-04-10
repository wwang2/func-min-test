---
strategy: basin-hopping
status: complete
eval_version: eval-v1
metric: 1.4995
issue: null
parents:
  - null
---

## Glossary

- **BH**: Basin Hopping — the global optimization algorithm implemented here
- **NM**: Nelder-Mead simplex — the local optimizer used inside each hop
- **SOTA**: State-of-the-art — the known best result for this benchmark
- **RNG**: Random Number Generator

## Approach

### The Problem

We minimize f(x,y) = sin(x)cos(y) + sin(xy) + (x²+y²)/20 over [-5, 5]².
The function is multi-modal: it has many local minima, but only one global minimum
at approximately (-1.704, 0.678) with value ~ -1.519.

The baseline random search achieves combined_score ≈ 1.438. It samples
uniformly at random and keeps the best — effective but wasteful.

### Why Basin Hopping?

The core insight of Basin Hopping (introduced by Wales & Doye, 1997) is to
transform the rugged energy landscape into a piecewise-constant "staircase"
surface where each point maps to the value of the local minimum in its basin.
This transformation removes kinetic barriers while preserving thermodynamic
information — the global minimum is still the lowest step on the staircase,
but random walks can reach it without being trapped by gradient barriers.

**The algorithm in three steps:**

1. **Descend.** From any point x, run a local optimizer (Nelder-Mead simplex)
   to find the basin minimum x*. Evaluate f(x*) — this is the "effective energy"
   at x on the transformed surface.

2. **Hop.** Perturb x* by a Gaussian step with standard deviation σ. Descend
   again to the new basin minimum x'*.

3. **Accept or reject** the new basin via the Metropolis criterion:
   accept always if f(x'*) < f(x*), else accept with probability
   exp(-(f(x'*) - f(x*)) / T), where T is a temperature parameter.

Adaptive step size: every 10 hops, if acceptance rate > 0.25 we increase σ
by 20% (steps are too small — not escaping basins); if < 0.25 we decrease σ
by 20% (steps are too large — landing in high-energy basins every time).

### Local Optimizer: Nelder-Mead Simplex

Implemented from scratch using only numpy. A simplex is a triangle (3 vertices)
in 2D. Each iteration evaluates which vertex is worst and moves it by
reflection, expansion, or contraction. Convergence when max |f_i - f_best| < tol.

Parameters: α=1.0 (reflect), γ=2.0 (expand), ρ=0.5 (contract), σ=0.5 (shrink).
These are the standard Nelder-Mead defaults (Nelder & Mead, 1965).

### Hyperparameters (final)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_hops | 200 | Generous budget; each hop takes ~5ms |
| n_starts | 8 | Multiple independent restarts ensure coverage |
| step_size | 1.2 | Initial σ — about 1/4 of domain width |
| temperature | 0.5 | Allows ~30% acceptance at ΔE=0.5 |
| target_accept | 0.25 | Standard BH practice |
| time_limit | 4.0s | Leave 1s margin vs 5s timeout |

## Results

### Evaluator runs (8 independent evaluations, 10 trials each)

| Run | combined_score | value_score | distance_score | reliability_score |
|-----|---------------|-------------|----------------|-------------------|
| 0   | 1.4995 | 0.9997 | 0.9995 | 1.0000 |
| 1   | 1.4995 | 0.9997 | 0.9995 | 1.0000 |
| 2   | 1.4995 | 0.9997 | 0.9995 | 1.0000 |
| 3   | 1.4995 | 0.9997 | 0.9995 | 1.0000 |
| 4   | 1.4995 | 0.9997 | 0.9995 | 1.0000 |
| 5   | 1.4995 | 0.9997 | 0.9995 | 1.0000 |
| 6   | 1.4995 | 0.9997 | 0.9995 | 1.0000 |
| 7   | 1.4995 | 0.9997 | 0.9995 | 1.0000 |
| **Mean** | **1.4995 ± 0.0000** | | | |

**Seeds (per evaluator run):** each of the 10 internal trials uses `np.random.randint(0, 2**31-1)` so seeds are stochastic across calls, ensuring diverse coverage.

### Comparison to baseline

| Method | combined_score | Improvement |
|--------|---------------|-------------|
| Random search (baseline) | 1.438 | — |
| Basin Hopping (ours) | 1.4995 ± 0.0000 | +4.3% |

The algorithm finds the global minimum at (-1.7041, 0.6775) with f ≈ -1.5187,
distance 0.0005 from the known global minimum. All 10 evaluator trials per run
converge to the same basin. Wall-clock time per `run_search()` call: ~120ms.

### Iteration history (3 variations tested)

| Variation | Change | combined_score | Notes |
|-----------|--------|---------------|-------|
| v1 | Initial: n_hops=60, n_starts=4 | ~1.454 ± 0.064 | Occasional misses |
| v2 | n_hops=200, n_starts=8, time_limit | ~1.499 ± 0.006 | Near-perfect |
| v3 | Final polish step (NM with tol=1e-12) | 1.4995 ± 0.0000 | Fully consistent |

## What Worked

- **Multiple restarts** (n_starts=8) were the biggest reliability improvement —
  they ensure global coverage even when individual chains get trapped.
- **Adaptive step size** prevents premature convergence and keeps the Markov
  chain mixing across basins.
- **Final high-precision polish** (NM with tol=1e-12) pushes the reported value
  to within 0.001 of the true global minimum, maximizing value_score.
- The algorithm is extremely fast (~120ms per call vs 5s budget), leaving room
  for many more hops if needed.

## What Didn't Work / Limitations

- Very high temperatures (T > 2.0) cause random-walk-like behavior with poor
  convergence — the Metropolis filter becomes too permissive.
- Very large step sizes (σ > 3.0) frequently land far from any useful basin.
- The algorithm could fail on higher-dimensional problems (d > 10) where NM
  simplex degrades, but for this 2D problem it is excellent.

## Prior Art & Novelty

### What is already known

- Basin Hopping was introduced by [Wales & Doye (1997)](https://doi.org/10.1021/jp970984n)
  for molecular cluster optimization and is a well-established global optimization method.
- The Nelder-Mead simplex method was published by [Nelder & Mead (1965)](https://doi.org/10.1093/comjnl/7.4.308).
- The Metropolis acceptance criterion comes from [Metropolis et al. (1953)](https://doi.org/10.1063/1.1699114).
- scipy.optimize.basinhopping implements this exact approach.

### What this orbit adds (if anything)

- This orbit applies known techniques (BH + NM) to the specific function
  minimization benchmark. No novelty claim is made.
- The custom NM implementation avoids the scipy dependency constraint.

### Honest positioning

This orbit applies well-known global optimization (Basin Hopping, Wales & Doye 1997)
to a 2D benchmark. The approach is a direct application of published methods —
there is no algorithmic novelty. The contribution is demonstrating that BH with
a pure-numpy NM local optimizer achieves near-optimal combined_score (1.4995)
reliably within the 5-second timeout, outperforming random search (1.438) by
approximately 4.3% under the evaluator's composite metric.

## References

- [Wales & Doye (1997)](https://doi.org/10.1021/jp970984n) — Basin Hopping algorithm for global optimization
- [Nelder & Mead (1965)](https://doi.org/10.1093/comjnl/7.4.308) — Simplex method for function minimization
- [Metropolis et al. (1953)](https://doi.org/10.1063/1.1699114) — Monte Carlo importance sampling / acceptance criterion

![Results figure](figures/results.png)
