"""Parallel evaluation script for diff-evo solution across multiple seeds."""

import sys
import os
import numpy as np
from multiprocessing import Pool

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "research", "eval")
)

SOLUTION_PATH = os.path.join(os.path.dirname(__file__), "solution.py")


def run_eval(_seed):
    from evaluator import evaluate

    result = evaluate(SOLUTION_PATH)
    return result.metrics["combined_score"]


if __name__ == "__main__":
    seeds = [42, 123, 7]
    import time

    t0 = time.perf_counter()
    with Pool(len(seeds)) as p:
        scores = p.map(run_eval, seeds)
    dt = time.perf_counter() - t0
    print(f"Scores: {scores}")
    print(f"Mean ± std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"Wall time: {dt:.1f}s")
