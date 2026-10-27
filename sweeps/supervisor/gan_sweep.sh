#!/bin/bash
# Runs each sweep entry (outputs in results/) in turn: "<out_dir> <seed> <json config overrides>".
cd /workspace
export PYTHONUNBUFFERED=1
while read -r out seed overrides; do
    [ -z "$out" ] && continue
    if [ -e "results/$out.zip" ]; then echo "=== $out: already done, skipping"; continue; fi
    echo "=== $out: seed $seed, overrides $overrides"
    /workspace/.venv/bin/python -c "
import json, sys
from workspace import gan
gan.CONFIG.update(json.loads(sys.argv[3]))
gan.run(out="results/" + sys.argv[1], seed=int(sys.argv[2]), show=False)" "$out" "$seed" "$overrides" 2>&1
    echo "=== $out: exit $?"
done < /workspace/sweeps/sweep.txt
echo "=== sweep finished"
