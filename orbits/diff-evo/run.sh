#!/usr/bin/env bash
# Reproduce the diff-evo orbit experiment
# Run from the repository root: bash orbits/diff-evo/run.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== diff-evo: Differential Evolution ==="
echo ""

# 1. Quick timing/accuracy check
echo "--- Single-call timing check (5 runs) ---"
python3 orbits/diff-evo/solution.py

echo ""

# 2. Full evaluation (10 trials per run)
echo "--- Full evaluator run ---"
python3 -c "
import sys
sys.path.insert(0, 'research/eval')
from evaluator import evaluate
r = evaluate('orbits/diff-evo/solution.py')
print('Metrics:', r.metrics)
print('Combined score:', r.metrics['combined_score'])
"

echo ""

# 3. Parallel evaluation across 3 seeds
echo "--- Parallel evaluation (3 seeds, should match 1.4995 ± ~0) ---"
python3 orbits/diff-evo/parallel_eval.py

echo ""

# 4. Generate results figure
echo "--- Generating figure ---"
python3 orbits/diff-evo/plot_results.py

echo ""
echo "=== Done. Results in orbits/diff-evo/figures/results.png ==="
