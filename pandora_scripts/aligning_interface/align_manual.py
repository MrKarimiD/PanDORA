import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

def capture_points_on_image(fig, ax, n):
    plt.sca(ax)
    pts = plt.ginput(n, timeout=0)

    for e in range(len(pts)):
        ax.add_patch(patches.Circle((pts[e][0], pts[e][1]), 1.0, edgecolor='r', facecolor='none'))
        ax.text(pts[e][0], pts[e][1], e, color = 'r')
    
    fig.canvas.draw()
    return pts

def align_manual(im, im_HDR, n):
    im_mod = im.copy()
    im_HDR_mod = im_HDR.copy()

    # Show images to select points
    fig, axes = plt.subplots(1, 2, figsize=(120, 120), gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axes[0].imshow(im_HDR_mod)
    axes[0].axis('off')
    axes[1].imshow(im_mod)
    axes[1].axis('off')
    
    # Manually select points
    im_HDR_pts = capture_points_on_image(fig, axes[0], n)
    im_pts = capture_points_on_image(fig, axes[1], n)
    
    # plt.savefig('points.png')
    plt.close()

    # Warp image 
    height, width = im.shape[:2]
    H, sattus = cv2.findHomography(np.float32(im_HDR_pts), np.float32(im_pts))
    warped_image = cv2.warpPerspective(im_HDR, H, (width, height))

    alpha = 50.0 / 100.0
    beta = 1.0 - alpha
    merged_image = cv2.addWeighted(im, beta, warped_image, alpha, 0)

    return warped_image, H