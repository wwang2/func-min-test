"""Run the evaluator 3 times sequentially (the evaluator itself runs 10 trials internally)."""

import sys
import time
import numpy as np

sys.path.insert(0, "research/eval")
from evaluator import evaluate

SOLUTION = "orbits/basin-hop/solution.py"
SEEDS = [42, 123, 7]

results = []
for seed in SEEDS:
    t0 = time.time()
    r = evaluate(SOLUTION)
    elapsed = time.time() - t0
    score = r.metrics.get("combined_score", r.combined_score)
    results.append(
        {"seed": seed, "score": score, "metrics": r.metrics, "elapsed": elapsed}
    )
    print(f"Seed {seed}: combined_score={score:.4f}  elapsed={elapsed:.1f}s")
    print(f"  metrics: {r.metrics}")

scores = [r["score"] for r in results]
print(f"\nMean ± Std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
