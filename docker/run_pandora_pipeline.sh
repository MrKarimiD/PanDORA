#!/usr/bin/env bash
#
# Fixed PanDORA two-stage training pipeline (README "Option A").
# Runs inside the PanDORA docker image, but also works in an interactive shell.
#
#   step 1 (lantern_steps 1)  ->  align_fast_to_well.py  ->  step 2 (lantern_steps 2)
#
# The dataset is provided via the DATASET_DIR env var (a processed *_ns scene).
# Deterministic --experiment-name / --timestamp make the step-1 outputs a known
# path that the align + step-2 commands can reference.
set -euo pipefail

# Make the repo root importable so `lantern_scripts` (used by nerfstudio's lantern
# processors/config) resolves even if PYTHONPATH isn't already set in the env.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# --- Configuration (override via environment) --------------------------------
# Processed nerfstudio scene directory (contains transforms.json, images/, ...).
DATASET_DIR="${DATASET_DIR:-/data/sample_scene/sample_scene_ns/}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pandora}"
# Iteration counts (lower these for a quick smoke test, e.g. STEP1_ITERS=200).
STEP1_ITERS="${STEP1_ITERS:-60000}"
STEP2_ITERS="${STEP2_ITERS:-120000}"
NUM_IMAGES_TO_SAMPLE="${NUM_IMAGES_TO_SAMPLE:-1800}"

STEP1_OUT="outputs/${EXPERIMENT_NAME}/PanDORA/step_1"
STEP2_OUT="outputs/${EXPERIMENT_NAME}/PanDORA/step_2"

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR does not exist: $DATASET_DIR"
    echo "Set DATASET_DIR to a processed nerfstudio scene (a *_ns directory)."
    exit 1
fi

echo "=================================================================="
echo "PanDORA two-stage training"
echo "  DATASET_DIR      = $DATASET_DIR"
echo "  EXPERIMENT_NAME  = $EXPERIMENT_NAME"
echo "  step 1 iters     = $STEP1_ITERS   -> $STEP1_OUT"
echo "  step 2 iters     = $STEP2_ITERS   -> $STEP2_OUT"
echo "=================================================================="

# --- Step 1: train on two exposures (lantern step 1) -------------------------
echo "[1/3] Training PanDORA step 1..."
ns-train PanDORA \
    --data "$DATASET_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --timestamp step_1 \
    --output-dir outputs \
    --pipeline.datamanager.train-num-images-to-sample-from "$NUM_IMAGES_TO_SAMPLE" \
    --pipeline.model.lantern_steps 1 \
    --pipeline.datamanager.pixel-sampler.lantern_steps 1 \
    --max-num-iterations "$STEP1_ITERS" \
    --viewer.quit-on-train-completion True

# --- Step 2a: align fast exposure to well exposure ---------------------------
echo "[2/3] Aligning fast exposure to well exposure..."
python lantern_scripts/align_fast_to_well.py \
    --input_dir "$DATASET_DIR" \
    --config "$STEP1_OUT/config.yml"

# --- Step 2b: train on aligned two exposures (lantern step 2) ----------------
echo "[3/3] Training PanDORA step 2..."
ns-train PanDORA \
    --data "$DATASET_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --timestamp step_2 \
    --output-dir outputs \
    --pipeline.datamanager.train-num-images-to-sample-from "$NUM_IMAGES_TO_SAMPLE" \
    --pipeline.model.lantern_steps 2 \
    --pipeline.datamanager.pixel-sampler.lantern_steps 2 \
    --pipeline.model.apply_mu_law False \
    --load-dir "$STEP1_OUT/nerfstudio_models" \
    --max-num-iterations "$STEP2_ITERS" \
    --viewer.quit-on-train-completion True

echo "=================================================================="
echo "Done. Final model: $STEP2_OUT"
echo "=================================================================="
