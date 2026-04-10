#!/bin/bash
# Reproduce the multi-start gradient descent evaluation
# Usage: cd <repo-root> && bash orbits/multistart-gd/run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

python3 -c "
import sys, numpy as np
sys.path.insert(0, 'research/eval')
from evaluator import evaluate

seeds = [42, 123, 7]
scores = []
for seed in seeds:
    np.random.seed(seed)
    r = evaluate('orbits/multistart-gd/solution.py')
    m = r.metrics
    scores.append(m['combined_score'])
    print(f'Seed {seed}: combined={m[\"combined_score\"]:.6f}  value={m[\"value_score\"]:.6f}  dist={m[\"distance_score\"]:.6f}  rel={m[\"reliability_score\"]:.1f}')

mean = np.mean(scores)
std = np.std(scores)
print(f'Mean: {mean:.6f} +/- {std:.6f}')
"
