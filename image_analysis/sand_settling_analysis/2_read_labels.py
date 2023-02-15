"""
Determine compaction of FluidFlower by comparing two different images.

The images correspond to the baseline image of the official well test
performed under the benchmark, and one of the other baseline images,
most likely close to C1. Between these two images, compaction/sedimentation
has occurred, i.e., to most degree the sand sunk from the src (well test)
to dst (C1 like) scenarios.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from src.io import read_paths_from_user_data
from src.largerigco2analysis import LargeRigCO2Analysis

# Loop through all runs
water_mask = {}
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

    water_mask[run] = fluidflower.labels == 0

# Store to file
Path("labels").mkdir(parents=True, exist_ok=True)
np.save("labels/c1_water.npy", water_mask["c1"])
np.save("labels/c3_water.npy", water_mask["c3"])

# Control
if True:
    plt.figure("c1 water")
    plt.imshow(water_mask["c1"])
    plt.figure("c3 water")
    plt.imshow(water_mask["c3"])
    plt.show()
