#!/usr/bin/env bash
# Run calculate_metrics_fast.py for every scene folder in a data root.
set -u

DATA_ROOT="/mnt/data/nerfstudio_ds/real_data"
SCRIPT="lantern_scripts/calculate_metrics_fast.py"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <metric_name> [data_root]"
    exit 1
fi

METRIC_NAME="$1"
DATA_ROOT="${2:-$DATA_ROOT}"

failed=()

for scene in "$DATA_ROOT"/*/; do
    [ -d "$scene" ] || continue
    echo "=================================================================="
    echo "Processing: $scene"
    echo "=================================================================="
    if python "$SCRIPT" --input_dir "$scene" --metric_name "$METRIC_NAME"; then
        echo "OK: $scene"
    else
        echo "FAILED: $scene"
        failed+=("$scene")
    fi
done

echo "=================================================================="
if [ ${#failed[@]} -eq 0 ]; then
    echo "All scenes processed successfully."
else
    echo "The following scenes failed:"
    for f in "${failed[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
