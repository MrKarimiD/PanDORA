#!/usr/bin/env bash
#
# Download PanDORA dataset scenes by name.
# Scenes are hosted as individual ZIP archives (~5 GB each) at:
#   https://hdrdb-public.s3.valeria.science/pandora/<scene>.zip
# See: https://lvsn.github.io/pandora/dataset/index.html
#
# Usage:
#   docker/download_scenes.sh <scene> [<scene> ...]
#   docker/download_scenes.sh all
#   docker/download_scenes.sh --list
#
# Options:
#   --dest DIR    Destination directory (default: $DATA_ROOT or /data)
#   --keep-zip    Keep the downloaded .zip after extracting (default: delete)
#   --force       Re-download / re-extract even if already present
#   --list        Print the available scene names and exit
#
# Downloads are resumable (curl -C -), so re-running continues an interrupted
# download. Extraction is skipped if the scene is already present unless --force.
set -euo pipefail

BASE_URL="https://hdrdb-public.s3.valeria.science/pandora"

# The 14 available scenes (exact zip basenames).
SCENES=(
    auditorium
    auditorium_dark
    basement
    blue_bedroom
    classroom_no_windows
    classroom_windows
    clubhouse
    coffee_room
    lab_office
    living_room
    lobby
    meeting_room
    office
    small_office
)

DEST="${DATA_ROOT:-/data}"
KEEP_ZIP=0
FORCE=0
REQUESTED=()

is_valid_scene() {
    local s="$1"
    for known in "${SCENES[@]}"; do
        [ "$known" = "$s" ] && return 0
    done
    return 1
}

print_list() {
    echo "Available PanDORA scenes (${#SCENES[@]}):"
    for s in "${SCENES[@]}"; do echo "  - $s"; done
}

# --- Parse arguments ---------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --dest)     DEST="$2"; shift 2 ;;
        --keep-zip) KEEP_ZIP=1; shift ;;
        --force)    FORCE=1; shift ;;
        --list)     print_list; exit 0 ;;
        -h|--help)  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        all)        REQUESTED=("${SCENES[@]}"); shift ;;
        -*)         echo "Unknown option: $1" >&2; exit 1 ;;
        *)          REQUESTED+=("$1"); shift ;;
    esac
done

if [ ${#REQUESTED[@]} -eq 0 ]; then
    echo "ERROR: no scene(s) specified." >&2
    echo >&2
    print_list >&2
    echo >&2
    echo "Example: docker/download_scenes.sh meeting_room coffee_room" >&2
    exit 1
fi

# Validate all requested names up front.
for s in "${REQUESTED[@]}"; do
    if ! is_valid_scene "$s"; then
        echo "ERROR: unknown scene '$s'." >&2
        echo >&2
        print_list >&2
        exit 1
    fi
done

mkdir -p "$DEST"

# --- Download + extract ------------------------------------------------------
for scene in "${REQUESTED[@]}"; do
    marker="$DEST/.$scene.done"
    zip="$DEST/$scene.zip"
    url="$BASE_URL/$scene.zip"

    echo "=================================================================="
    echo "Scene: $scene"
    echo "  URL : $url"
    echo "  Dest: $DEST"
    echo "=================================================================="

    if [ -f "$marker" ] && [ "$FORCE" -eq 0 ]; then
        echo "Already downloaded and extracted (marker $marker exists); skipping."
        echo "Use --force to re-download."
        continue
    fi

    echo "Downloading (resumable)..."
    curl -L -f --retry 5 --retry-delay 5 -C - -o "$zip" "$url"

    echo "Extracting..."
    unzip -q -o "$zip" -d "$DEST"

    touch "$marker"

    if [ "$KEEP_ZIP" -eq 0 ]; then
        echo "Removing archive $zip (use --keep-zip to keep)..."
        rm -f "$zip"
    fi

    echo "Done: $scene"
done

echo "=================================================================="
echo "All requested scenes are ready under: $DEST"
echo "=================================================================="
