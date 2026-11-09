"""Align images from one folder to a reference folder using the same
contour-matching + homography computation as align_fast_to_well.py, but
WITHOUT needing a model to render the reference images.

It loads two folders of already-existing images:
  --well_dir : reference images to align *to* (previously the rendered ones)
  --fast_dir : images to be warped/aligned
and, for each matched pair, computes the homography and saves the warped image.
"""

import cv2
import numpy as np
import os
import sys
import argparse

# Reuse the alignment logic from the original script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_fast_to_well import AlignImages, save_transformation


IMAGE_EXTS = ('.png', '.jpg', '.jpeg')


def list_images(directory, prefix=""):
    files = [f for f in os.listdir(directory)
             if f.lower().endswith(IMAGE_EXTS) and f.startswith(prefix)]
    return sorted(files)


def pair_images(well_files, fast_files, match_by):
    """Return a list of (well_filename, fast_filename) pairs."""
    if match_by == 'name':
        well_by_stem = {os.path.splitext(f)[0]: f for f in well_files}
        pairs = []
        for fast_f in fast_files:
            stem = os.path.splitext(fast_f)[0]
            if stem in well_by_stem:
                pairs.append((well_by_stem[stem], fast_f))
            else:
                print(f"No matching well image for: {fast_f}")
        return pairs
    else:  # 'order'
        if len(well_files) != len(fast_files):
            print(f"Warning: folder counts differ "
                  f"(well={len(well_files)}, fast={len(fast_files)}); "
                  f"pairing the first {min(len(well_files), len(fast_files))} by sorted order.")
        return list(zip(well_files, fast_files))


def save_checkerboard(a, b, path, tiles=8):
    """Alternating tiles of two RGB images to expose misaligned edges."""
    h, w = a.shape[:2]
    th, tw = h // tiles, w // tiles
    out = a.copy()
    for i in range(tiles):
        for j in range(tiles):
            if (i + j) % 2 == 1:
                y0, y1 = i * th, (h if i == tiles - 1 else (i + 1) * th)
                x0, x1 = j * tw, (w if j == tiles - 1 else (j + 1) * tw)
                out[y0:y1, x0:x1] = b[y0:y1, x0:x1]
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def save_matched_contours(align_images, well_path, fast_path):
    """Save the matched contours on the well and fast images as two separate
    images. Each matched pair uses the same color and index in both, so they
    can be cross-referenced."""
    well = align_images.well_image.image.copy()
    fast = align_images.fast_image.image.copy()

    pairs = list(zip(align_images.matched_well_contours, align_images.fast_contours))

    def draw(canvas, contour, idx, color):
        cv2.drawContours(canvas, [contour.contour], -1, color, 2)
        cx, cy = int(contour.centroid[0]), int(contour.centroid[1])
        cv2.putText(canvas, str(idx), (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for idx, (well_c, fast_c) in enumerate(pairs):
        if well_c is None or fast_c is None:
            continue
        # Deterministic distinct color per matched pair (same in both images).
        color_map = cv2.applyColorMap(
            np.array([[int(255 * idx / max(len(pairs), 1))]], dtype=np.uint8),
            cv2.COLORMAP_HSV)[0, 0]
        color = (int(color_map[0]), int(color_map[1]), int(color_map[2]))
        draw(well, well_c, idx, color)
        draw(fast, fast_c, idx, color)

    cv2.imwrite(well_path, cv2.cvtColor(well, cv2.COLOR_RGB2BGR))
    cv2.imwrite(fast_path, cv2.cvtColor(fast, cv2.COLOR_RGB2BGR))


def _rotate_ccw_point(x, y, width):
    """Map a point (x, y) under a 90 deg counter-clockwise image rotation,
    where `width` is the ORIGINAL image width."""
    return y, (width - 1 - x)


def _rotate_ccw_contour(contour, width):
    """Apply the 90 deg CCW point mapping to an (N,1,2) contour array."""
    pts = contour.reshape(-1, 2)
    rotated = np.empty_like(pts)
    rotated[:, 0] = pts[:, 1]
    rotated[:, 1] = width - 1 - pts[:, 0]
    return rotated.reshape(-1, 1, 2)


def save_matched_contours_panel(align_images, path):
    """Side-by-side panel with each image rotated 90 deg counter-clockwise,
    each matched contour pair drawn in the same color and connected by a line
    across the panel."""
    well = cv2.rotate(align_images.well_image.image.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
    fast = cv2.rotate(align_images.fast_image.image.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Original widths (needed for the coordinate rotation).
    w_w0 = align_images.well_image.image.shape[1]
    w_f0 = align_images.fast_image.image.shape[1]
    # Rotated image dimensions.
    h_w, w_w = well.shape[:2]
    h_f, w_f = fast.shape[:2]
    h = max(h_w, h_f)
    canvas = np.zeros((h, w_w + w_f, 3), dtype=np.uint8)
    canvas[:h_w, :w_w] = well
    canvas[:h_f, w_w:w_w + w_f] = fast

    pairs = list(zip(align_images.matched_well_contours, align_images.fast_contours))
    for idx, (well_c, fast_c) in enumerate(pairs):
        if well_c is None or fast_c is None:
            continue
        # Deterministic distinct color per matched pair.
        color_map = cv2.applyColorMap(
            np.array([[int(255 * idx / max(len(pairs), 1))]], dtype=np.uint8),
            cv2.COLORMAP_HSV)[0, 0]
        color = (int(color_map[0]), int(color_map[1]), int(color_map[2]))

        well_contour_rot = _rotate_ccw_contour(well_c.contour, w_w0)
        fast_contour_rot = _rotate_ccw_contour(fast_c.contour, w_f0)
        cv2.drawContours(canvas, [well_contour_rot], -1, color, 2)
        cv2.drawContours(canvas, [fast_contour_rot + np.array([[w_w, 0]])], -1, color, 2)

        wx, wy = _rotate_ccw_point(int(well_c.centroid[0]), int(well_c.centroid[1]), w_w0)
        fx, fy = _rotate_ccw_point(int(fast_c.centroid[0]), int(fast_c.centroid[1]), w_f0)
        fx += w_w
        cv2.line(canvas, (wx, wy), (fx, fy), color, 3)

    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_sift_matches_panel(align_images, path, layout="vertical", max_matches=40):
    """Panel drawing SIFT/BFMatcher point correspondences (post-BFMatcher,
    pre-RANSAC) as connecting lines.
    layout='vertical'   -> fast exposure on top, well exposure on bottom.
    layout='horizontal' -> fast exposure on left, well exposure on right.
    max_matches limits how many correspondences are drawn (evenly subsampled)
    so the visualization stays readable."""
    well = align_images.well_image.image.copy()
    fast = align_images.fast_image.image.copy()
    h_w, w_w = well.shape[:2]
    h_f, w_f = fast.shape[:2]

    if layout == "horizontal":
        canvas = np.zeros((max(h_w, h_f), w_f + w_w, 3), dtype=np.uint8)
        canvas[:h_f, :w_f] = fast                  # fast exposure on left
        canvas[:h_w, w_f:w_f + w_w] = well         # well exposure on right
        fast_offset = (0, 0)
        well_offset = (w_f, 0)
    else:  # vertical
        canvas = np.zeros((h_f + h_w, max(w_w, w_f), 3), dtype=np.uint8)
        canvas[:h_f, :w_f] = fast                  # fast exposure on top
        canvas[h_f:h_f + h_w, :w_w] = well         # well exposure on bottom
        fast_offset = (0, 0)
        well_offset = (0, h_f)

    well_pts = align_images.well_points_combined
    fast_pts = align_images.fast_points_combined
    n = min(len(well_pts), len(fast_pts))

    # Evenly subsample down to max_matches for readability.
    if max_matches and n > max_matches:
        indices = np.linspace(0, n - 1, max_matches).astype(int)
    else:
        indices = range(n)

    for k, idx in enumerate(indices):
        wp = well_pts[idx]
        fp = fast_pts[idx]
        # Distinct color per drawn correspondence.
        color_map = cv2.applyColorMap(
            np.array([[int(255 * k / max(len(indices), 1))]], dtype=np.uint8),
            cv2.COLORMAP_HSV)[0, 0]
        color = (int(color_map[0]), int(color_map[1]), int(color_map[2]))
        fx, fy = int(fp[0]) + fast_offset[0], int(fp[1]) + fast_offset[1]
        wx, wy = int(wp[0]) + well_offset[0], int(wp[1]) + well_offset[1]
        cv2.circle(canvas, (fx, fy), 5, color, -1)
        cv2.circle(canvas, (wx, wy), 5, color, -1)
        cv2.line(canvas, (fx, fy), (wx, wy), color, 3)

    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_thresholded(align_images, output_dir, base):
    """Save the binary threshold maps that contour detection operates on."""
    cv2.imwrite(os.path.join(output_dir, base + "_fast_threshold.png"),
                align_images.fast_image.thresholded)
    for img, thr in zip(align_images.well_images, align_images.well_thresholds):
        cv2.imwrite(os.path.join(output_dir, base + f"_well_threshold_{thr}.png"),
                    img.thresholded)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--well_dir", type=str, required=True,
                           help="Folder of reference images to align to.")
    argparser.add_argument("--fast_dir", type=str, required=True,
                           help="Folder of images to be aligned (warped).")
    argparser.add_argument("--output_dir", type=str, required=True,
                           help="Folder to write the aligned images into.")
    argparser.add_argument("--match_by", type=str, default="order",
                           choices=["order", "name"],
                           help="Pair images by sorted order (default) or by filename stem.")
    argparser.add_argument("--prefix", type=str, default="right",
                           help="Only align fast_dir images whose filename starts with this prefix "
                                "(use '' to process all images).")
    argparser.add_argument("--stride", type=int, default=1,
                           help="Process only every Nth image pair (e.g. 5 = every 5th).")
    argparser.add_argument("--only", type=str, default=None,
                           help="Process only this single fast_dir filename (its correct "
                                "well partner is still resolved via the full pairing).")
    argparser.add_argument("--save_merged", action='store_true', default=False,
                           help="Also save before/after overlay visualizations.")
    argparser.add_argument("--visualize_contours", action='store_true', default=False,
                           help="Save detected/matched contour visualizations for debugging.")
    argparser.add_argument("--save_diagnostics", action='store_true', default=False,
                           help="Save diagnostics: before/after checkerboard composites, "
                                "thresholded binary maps, and contour visualizations.")
    argparser.add_argument("--sift_layout", type=str, default="vertical",
                           choices=["vertical", "horizontal"],
                           help="Layout for *_sift_matches.png: vertical (fast top / well bottom) "
                                "or horizontal (fast left / well right).")
    argparser.add_argument("--sift_max_matches", type=int, default=40,
                           help="Max number of SIFT correspondences drawn in *_sift_matches.png "
                                "(evenly subsampled for readability; 0 = draw all).")
    args = argparser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, 'alignment_matrices.json')
    merged_dir = os.path.join(args.output_dir, 'merged')
    contour_visualizations_path = os.path.join(args.output_dir, 'contour_visualizations')
    diagnostics_dir = os.path.join(args.output_dir, 'diagnostics')
    if args.save_merged:
        os.makedirs(merged_dir, exist_ok=True)
    if args.save_diagnostics:
        os.makedirs(diagnostics_dir, exist_ok=True)

    well_files = list_images(args.well_dir)
    fast_files = list_images(args.fast_dir, prefix=args.prefix)
    pairs = pair_images(well_files, fast_files, args.match_by)

    if args.only is not None:
        pairs = [(w, f) for (w, f) in pairs if f == args.only]
        if not pairs:
            print(f"'{args.only}' not found among the paired fast images; nothing to do.")
            sys.exit(0)

    if args.stride > 1:
        pairs = pairs[::args.stride]

    print(f"Aligning {len(pairs)} image pairs...")
    for well_name, fast_name in pairs:
        well_path = os.path.join(args.well_dir, well_name)
        fast_path = os.path.join(args.fast_dir, fast_name)
        well_image = cv2.cvtColor(cv2.imread(well_path), cv2.COLOR_BGR2RGB)
        fast_image = cv2.cvtColor(cv2.imread(fast_path), cv2.COLOR_BGR2RGB)

        align_images = None
        try:
            align_images = AlignImages(well_image, fast_image)
            warped_image, matrix = align_images.alignImages()
            print(f"Aligned: {fast_name} -> {well_name}")
        except Exception as e:
            warped_image = fast_image
            matrix = np.zeros((3, 3))
            print(f"Failed to align: {fast_name} ({e})")

        base = os.path.splitext(fast_name)[0]

        if args.visualize_contours and align_images is not None:
            try:
                align_images.save_contour_visualizations(contour_visualizations_path, base + ".png")
            except Exception:
                print(f"Failed to visualize contours: {fast_name}")

        out_path = os.path.join(args.output_dir, fast_name)
        cv2.imwrite(out_path, cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))
        save_transformation(json_path, fast_name, matrix)

        if args.save_diagnostics:
            try:
                save_checkerboard(well_image, fast_image,
                                  os.path.join(diagnostics_dir, base + "_checkerboard_before.png"))
                save_checkerboard(well_image, warped_image,
                                  os.path.join(diagnostics_dir, base + "_checkerboard_after.png"))
                if align_images is not None:
                    save_thresholded(align_images, diagnostics_dir, base)
                    align_images.save_contour_visualizations(diagnostics_dir, base + ".png")
                    save_matched_contours(
                        align_images,
                        os.path.join(diagnostics_dir, base + "_matched_well.png"),
                        os.path.join(diagnostics_dir, base + "_matched_fast.png"))
                    save_matched_contours_panel(
                        align_images,
                        os.path.join(diagnostics_dir, base + "_matched_panel.png"))
                    save_sift_matches_panel(
                        align_images,
                        os.path.join(diagnostics_dir, base + "_sift_matches.png"),
                        layout=args.sift_layout,
                        max_matches=args.sift_max_matches)
            except Exception as e:
                print(f"Failed to save diagnostics: {fast_name} ({e})")

        if args.save_merged:
            alpha = 50.0 / 100.0
            merged_before = cv2.addWeighted(well_image, 1.0 - alpha, fast_image, alpha, 0)
            merged_before_path = os.path.join(merged_dir, fast_name).rsplit('.', 1)[0] + "_before.png"
            cv2.imwrite(merged_before_path, cv2.cvtColor(merged_before, cv2.COLOR_RGB2BGR))

            merged_after = cv2.addWeighted(well_image, 1.0 - alpha, warped_image, alpha, 0)
            merged_after_path = os.path.join(merged_dir, fast_name).rsplit('.', 1)[0] + "_after.png"
            cv2.imwrite(merged_after_path, cv2.cvtColor(merged_after, cv2.COLOR_RGB2BGR))

    print("All done :D")
