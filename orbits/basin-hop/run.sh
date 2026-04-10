#!/usr/bin/env bash
# Reproduce the basin-hop orbit experiment from the repository root.
# Usage: bash orbits/basin-hop/run.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== Basin Hopping: solution sanity test ==="
python3 orbits/basin-hop/solution.py

echo ""
echo "=== Full evaluator run (8 repetitions) ==="
python3 - <<'EOF'
import sys, numpy as np
sys.path.insert(0, 'research/eval')
from evaluator import evaluate

scores = []
for i in range(8):
    r = evaluate('orbits/basin-hop/solution.py')
    score = r.metrics.get('combined_score', 0)
    scores.append(score)
    print(f"Run {i}: combined_score={score:.4f}  "
          f"value_score={r.metrics.get('value_score',0):.4f}  "
          f"dist_score={r.metrics.get('distance_score',0):.4f}  "
          f"reliability={r.metrics.get('reliability_score',0):.4f}")

print(f"\nMean={np.mean(scores):.4f}  Std={np.std(scores):.4f}  Min={np.min(scores):.4f}")
EOF

echo ""
echo "=== Regenerate figure ==="
python3 orbits/basin-hop/plot_results.py

echo ""
echo "Done. Figure: orbits/basin-hop/figures/results.png"
