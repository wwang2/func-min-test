---
strategy: differential-evolution-nelder-mead
status: complete
eval_version: standalone
metric: 1.4995
issue: 2
parents:
  - null
---

## Glossary

- **DE**: Differential Evolution -- a population-based stochastic global optimizer that maintains a population of candidate solutions and iteratively improves them through mutation, crossover, and selection.
- **NM**: Nelder-Mead -- a simplex-based local optimizer that refines a solution without requiring gradient information.

## Results

| Evaluator Run | combined_score | value_score | distance_score | reliability_score |
|---------------|---------------|-------------|----------------|-------------------|
| 1             | 1.4995        | 0.9997      | 0.9995         | 1.0               |
| 2             | 1.4995        | 0.9997      | 0.9995         | 1.0               |
| 3             | 1.4995        | 0.9997      | 0.9995         | 1.0               |
| **Mean**      | **1.4995 +/- 0.0000** | | | |

Theoretical maximum: 1.5 (all sub-scores = 1.0, with the 1.5x quality multiplier for avg_distance < 0.5). We achieve 99.97% of the theoretical maximum.

**Baseline comparison:** Random search scored ~1.058 (1.2x multiplier, mediocre distance/value scores). Our approach scores 1.4995, a **42% relative improvement**.

## Approach

The objective function f(x,y) = sin(x)*cos(y) + sin(x*y) + (x^2+y^2)/20 is multi-modal on [-5,5]^2. The global minimum sits at approximately (-1.704, 0.678) with value -1.519. There are at least two significant local minima: near (1.584, 2.939) and (0.717, -2.487).

The core difficulty is that a local optimizer started from a random point has roughly a 1-in-3 chance of converging to a local minimum instead of the global one. Random search (the baseline) partially addresses this by sampling 1000 points uniformly, but it lacks the precision of a gradient-free local optimizer -- it finds the right basin but not the exact bottom.

**Solution: Multi-restart Differential Evolution + Nelder-Mead polish.**

1. **Differential Evolution (DE)** with `best1bin` strategy, population of 15, up to 100 generations, Latin hypercube initialization. Each run takes ~15-20ms and finds the global basin with ~86% probability.

2. **Nelder-Mead (NM) polish** on the DE result, with very tight tolerances (xatol=1e-12, fatol=1e-12). This refines the DE result from ~4 significant digits to ~10 significant digits of the true minimum location.

3. **10 independent restarts**, keeping the best result. The probability of all 10 missing the global basin is 0.14^10 < 3e-9. Total wall-clock time: ~200ms, well within the 5-second budget.

## What Happened

- **Iteration 1:** Single DE+NM run. Score = 1.364. The 1.5x multiplier was achieved (avg_distance < 0.5) but individual trials occasionally converged to local minima, dragging down averages.
- **Iteration 2:** Added 10 independent restarts. Score = 1.4995. All 10 evaluator trials now converge to the global minimum with high precision (distance < 0.001). Perfectly stable across repeated evaluations.

## What I Learned

- Differential Evolution with default-ish settings has ~86% per-run reliability on this function. That means 14% of individual runs land in local minima -- too much noise for a reliable evaluator score.
- Multi-restart is the right strategy when individual runs are cheap (~20ms each). 10 restarts costs only 200ms total but pushes failure probability below 1 in a billion.
- Nelder-Mead polish is essential for precision. DE alone gives ~4 digits of accuracy; NM pushes to ~10 digits. This matters for the distance_score sub-metric.
- The 1.5x quality multiplier (avg_distance < 0.5) is the dominant factor in the combined_score formula. Any approach that doesn't achieve this is leaving 50% performance on the table.

## Prior Art & Novelty

### What is already known
- [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) was introduced by Storn and Price (1997). It is a well-established global optimizer.
- [Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) dates to 1965. It is a standard local optimizer.
- Multi-start strategies combining global and local search are textbook approaches (see e.g., Marti 2003, "Multi-start methods").

### What this orbit adds
- This orbit applies known techniques to the specific evaluation problem. No novelty claim.
- The contribution is engineering: choosing the right parameters (popsize, maxiter, restart count) to maximize the evaluator's combined_score within the 5-second time budget.

### Honest positioning
This is a straightforward application of well-known optimization techniques. The only "tuning" was sizing the restart count to balance reliability against the time budget. Any optimization practitioner would reach the same solution.

## References

- Storn, R. and Price, K. (1997). "Differential Evolution -- A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces." Journal of Global Optimization, 11, 341-359.
- Nelder, J.A. and Mead, R. (1965). "A Simplex Method for Function Minimization." Computer Journal, 7(4), 308-313.
- [scipy.optimize.differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
- [scipy.optimize.minimize (Nelder-Mead)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
