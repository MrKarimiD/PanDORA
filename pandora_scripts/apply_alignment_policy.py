"""Build a stage-two dataset variant that filters badly aligned fast (right) views.

Run after align_fast_to_well.py, which writes the warped right views to images/ and masks/,
keeps the originals in images_unaligned/ and masks_unaligned/, and writes alignment_report.csv.

A view is flagged as a bad alignment (default --flag center) when its homography is
  - degenerate: |det of the top-left 2x2| < 0.5 or > 2 (includes reflections, det < 0), or a
    projective weight w <= 0 at the image centre or a corner; or
  - non-degenerate but moves the image centre by more than median + k * sigma_MAD
    (sigma_MAD = 1.4826 * MAD), computed over the non-degenerate views of this scene
    (or pass --threshold_px to use a fixed / pooled threshold in pixels).
--flag translation reproduces the first version: |(H02, H12)| above median + k * sigma_MAD.
That measure is the motion of the image corner, so small rotations/scales inflate it; avoid it.

Policies for flagged views:
    warp_all           keep the warped view (same data as align_fast_to_well.py's output)
    drop_flagged       keep the image but write all-zero masks, so stage two never samples it
    unaligned_flagged  use the original, unaligned image and masks

The variant is written to --output_dir as a lightweight copy of --input_dir: every file is a
symlink to the original, except the masks of dropped views, which are written as real files.
The original scene folder is never modified.

Example:
    python pandora_scripts/apply_alignment_policy.py --input_dir $DATA_PATH/lobby/lobby_ns \
        --output_dir $DATA_PATH/lobby/lobby_ns_drop --policy drop_flagged
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

POLICIES = ("warp_all", "drop_flagged", "unaligned_flagged")


def mad_threshold(values, k):
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    sigma = 1.4826 * np.median(np.abs(values - median))
    return float(median + k * sigma), float(median), float(sigma)


def load_homographies(input_dir):
    with open(os.path.join(input_dir, "alignment_matrices.json")) as f:
        return {item["name"]: np.asarray(item["matrix"], dtype=np.float64) for item in json.load(f)["matrices"]}


def homography_stats(H, width, height):
    """Returns (degenerate, centre displacement in px, det of the 2x2)."""
    H = H / H[2, 2]
    det = float(np.linalg.det(H[:2, :2]))
    pts = np.array([[width / 2, height / 2, 1], [0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]], dtype=np.float64)
    mapped = pts @ H.T
    w = mapped[:, 2]
    degenerate = (abs(det) < 0.5) or (abs(det) > 2.0) or bool(np.any(w <= 1e-9))
    if w[0] <= 1e-9:
        return True, float("inf"), det
    centre = mapped[0, :2] / w[0]
    return degenerate, float(np.hypot(centre[0] - width / 2, centre[1] - height / 2)), det


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def write_zero_mask(like_path, dst):
    mask = cv2.imread(like_path, cv2.IMREAD_GRAYSCALE)
    if os.path.lexists(dst):
        os.remove(dst)
    cv2.imwrite(dst, np.zeros_like(mask))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Nerfstudio folder of the scene, after align_fast_to_well.py")
    parser.add_argument("--output_dir", required=True, help="Where to build the variant (must not exist yet)")
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--k", type=float, default=3.0, help="Number of sigma_MAD above the median")
    parser.add_argument("--threshold_px", type=float, default=None, help="Use this threshold instead of computing it from the scene")
    parser.add_argument("--flag", choices=("center", "translation"), default="center",
                        help="center: degenerate H or centre-displacement outlier (default). translation: old |(H02,H12)| rule")
    parser.add_argument("--image_size", type=int, nargs=2, default=(960, 960), metavar=("W", "H"))
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    if os.path.exists(output_dir):
        raise SystemExit(f"{output_dir} already exists, remove it or pick another name")
    for needed in ("alignment_report.csv", "images", "masks", "images_unaligned", "masks_unaligned"):
        if not os.path.exists(os.path.join(input_dir, needed)):
            raise SystemExit(f"Missing {needed} in {input_dir}; run align_fast_to_well.py first")

    with open(os.path.join(input_dir, "alignment_report.csv")) as f:
        rows = list(csv.DictReader(f))
    aligned = [r for r in rows if r["status"] == "aligned" and r["translation_px"] != ""]
    num_degenerate = None
    if args.flag == "translation":
        translations = [float(r["translation_px"]) for r in aligned]
        if args.threshold_px is not None:
            threshold, median, sigma = args.threshold_px, None, None
        else:
            threshold, median, sigma = mad_threshold(translations, args.k)
        flagged = sorted(r["name"] for r in aligned if float(r["translation_px"]) > threshold)
    else:
        homographies = load_homographies(input_dir)
        stats = {r["name"]: homography_stats(homographies[r["name"]], *args.image_size) for r in aligned}
        degenerate = [n for n, (deg, _, _) in stats.items() if deg]
        centres = {n: c for n, (deg, c, _) in stats.items() if not deg}
        if args.threshold_px is not None:
            threshold, median, sigma = args.threshold_px, None, None
        else:
            threshold, median, sigma = mad_threshold(list(centres.values()), args.k)
        num_degenerate = len(degenerate)
        flagged = sorted(set(degenerate) | {n for n, c in centres.items() if c > threshold})

    # Lightweight copy: symlink every top-level entry except images/ and masks/.
    os.makedirs(output_dir)
    for entry in os.listdir(input_dir):
        if entry in ("images", "masks"):
            continue
        link(os.path.join(input_dir, entry), os.path.join(output_dir, entry))
    for folder in ("images", "masks"):
        os.makedirs(os.path.join(output_dir, folder))
        for entry in os.listdir(os.path.join(input_dir, folder)):
            link(os.path.join(input_dir, folder, entry), os.path.join(output_dir, folder, entry))

    if args.policy != "warp_all":
        for name in flagged:
            mask_names = [name, name.replace(".png", "_saturation_mask.png")]
            if args.policy == "unaligned_flagged":
                link(os.path.join(input_dir, "images_unaligned", name), os.path.join(output_dir, "images", name))
                for m in mask_names:
                    link(os.path.join(input_dir, "masks_unaligned", m), os.path.join(output_dir, "masks", m))
            else:  # drop_flagged
                for m in mask_names:
                    write_zero_mask(os.path.join(input_dir, "masks", m), os.path.join(output_dir, "masks", m))

    summary = {
        "policy": args.policy,
        "flag": args.flag,
        "num_degenerate": num_degenerate,
        "input_dir": input_dir,
        "k": args.k,
        "threshold_px": threshold,
        "median_px": median,
        "sigma_mad_px": sigma,
        "num_right_views": len(rows),
        "num_aligned": len(aligned),
        "num_no_lights": sum(r["status"] == "no_lights" for r in rows),
        "num_failed": sum(r["status"] == "failed" for r in rows),
        "num_flagged": len(flagged),
        "flagged_percent_of_aligned": 100.0 * len(flagged) / max(len(aligned), 1),
        "flagged_percent_of_all": 100.0 * len(flagged) / max(len(rows), 1),
        "flagged": flagged,
    }
    with open(os.path.join(output_dir, "alignment_policy.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "flagged"}, indent=2))
