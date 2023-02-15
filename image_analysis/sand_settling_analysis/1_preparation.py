"""
1. Step of compaction analysis for the FluidFlower.
"""

from pathlib import Path

import cv2
import numpy as np
import skimage
from src.io import read_paths_from_user_data
from src.largerigco2analysis import LargeRigCO2Analysis

# ! ---- Read baseline images of all runs

corrected_images = {}
# Loop through all runs
for run in ["c1", "c2", "c3", "c4", "c5"]:

    # Read user-defined paths to images etc.
    images, baseline, _, results = read_paths_from_user_data("../data.json", run)

    config = "../phase_segmentation/config.json"

    # Define FluidFlower based on a full set of basline images
    fluidflower = LargeRigCO2Analysis(
        baseline=[baseline[0]],  # paths to baseline images
        config=config,  # path to config file
        results=results,  # path to results directory
    )

    corrected_images[run] = fluidflower.base.img

# ! ---- Correct images 3, 4, 5 due to very slight camera movement.
h, w = corrected_images["c3"].shape[:2]

# X,Y pixels measured in inkscape (0, 0) is ine the lower left corner.
marks_pixels_c2 = np.array(
    [
        # Main marks (low uncertainty)
        [14.5, 24.5],  # White mark on the lower left corner
        # [16.5, 1526.5],  # White mark on the mid of the left frame
        # [218, 4000], # White mark on the color checker
        [7898.5, 28.5],  # Sand grain on the right lower corner
        [2738.5, 3181.5],  # White spot in the top of the fault seal
        [
            6912.5,
            3231.5,
        ],  # White spot in th eupper ESF sand - not changing for c1-2 and c3-5.
    ]
)

# Same for C4 and C5
marks_pixels_c3 = np.array(
    [
        [15.5, 24.5],
        # [17.5, 1526.5],
        # [233, 3843.5],
        [7895.5, 29.5],
        [2737.5, 3180.5],
        [6909.5, 3231.5],
    ]
)

# Convert to reverse pixel coordinates
marks_pixels_c2[:, 1] = h - marks_pixels_c2[:, 1]
marks_pixels_c3[:, 1] = h - marks_pixels_c3[:, 1]

# Find affine map to map c3 onto c2.
transformation, mask = cv2.findHomography(
    marks_pixels_c3, marks_pixels_c2, method=cv2.RANSAC
)

# Apply translation - Change the source and return it
for run in ["c3", "c4", "c5"]:
    corrected_images[run] = cv2.warpPerspective(
        skimage.img_as_float(corrected_images[run]).astype(np.float32),
        transformation.astype(np.float32),
        (w, h),
    )

# ! ----- Preliminaries - prepare two images for compaction analysis

# Store images unmodifed as npy arrays
Path("images").mkdir(parents=True, exist_ok=True)
for run in ["c1", "c2", "c3", "c4", "c5"]:
    np.save(f"images/{run}.npy", corrected_images[run])
