#!/bin/bash
# Reproduce the basin-hopping experiment
# Usage: cd <repo-root> && bash orbits/basin-hopping-scipy/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Basin-Hopping Scipy Optimizer ==="
echo "Running solution standalone..."
python3 "$SCRIPT_DIR/solution.py"

echo ""
echo "=== Running evaluator ==="
python3 -c "
import sys; sys.path.insert(0, '$REPO_ROOT/research/eval')
from evaluator import evaluate
r = evaluate('$SCRIPT_DIR/solution.py')
print(r)
print()
for k, v in r.metrics.items():
    print(f'  {k}: {v}')
"
