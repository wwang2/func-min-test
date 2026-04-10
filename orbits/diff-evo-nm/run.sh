#!/usr/bin/env bash
# Reproduce the diff-evo-nm orbit evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Evaluating Differential Evolution + Nelder-Mead solution ==="
python3 -c "
import sys
sys.path.insert(0, '${REPO_ROOT}/research/eval')
from evaluator import evaluate

scores = []
for i in range(3):
    r = evaluate('${SCRIPT_DIR}/solution.py')
    score = r.metrics['combined_score']
    scores.append(score)
    print(f'Run {i+1}: combined_score={score:.6f}')
    for k, v in r.metrics.items():
        if k != 'combined_score':
            print(f'  {k}: {v:.6f}')

import numpy as np
print(f'\nFinal: {np.mean(scores):.6f} +/- {np.std(scores):.6f}')
"
