"""
Script for visually comparing phase segmentation of different runs.

"""
import json
from datetime import datetime
from pathlib import Path

import darsia
import numpy as np
import pandas as pd
from src.io import read_paths_from_user_data, read_time_from_path
from src.largerigco2analysis import LargeRigCO2Analysis

from whole_image_analysis import whole_img

# ! ---- Folder for main results
# Fetch folder for storing the results
config_path = Path("../data.json")
f = open(config_path, "r")
config = json.load(f)
f.close()
main_results = Path(config["main"]["results"])
main_results = main_results / Path("comp_images_weighted/")
main_results.mkdir(parents=True, exist_ok=True)

# ! ----  Injection times for C1, ..., C5.

inj_start_times = {
    "c1": datetime(2021, 11, 24, 8, 31, 0),
    "c2": datetime(2021, 12, 4, 10, 1, 0),
    "c3": datetime(2021, 12, 14, 11, 20, 0),
    "c4": datetime(2021, 12, 24, 9, 0, 0),
    "c5": datetime(2022, 1, 4, 11, 0, 0),
}

# Fetch path to segmentation data
seg_folders = {}
segmentations = {}
for run in ["c1", "c2", "c3", "c4", "c5"]:
    # Read user-defined paths to images etc.
    images, baseline, _, results = read_paths_from_user_data("../data.json", run)
    seg_folder = results / Path("npy_segmentation")
    segmentations[run] = list(sorted(seg_folder.glob("*.npy")))

    # Define baseline image (need to correct for geometrical distortion)
    if run == "c1":
        config = "../phase_segmentation/config.json"
        # Define FluidFlower based on a full set of basline images
        co2_analysis = LargeRigCO2Analysis(
            baseline=[baseline[0]],  # paths to baseline images
            config=config,  # path to config file
            results=results,  # path to results directory
        )
        background: darsia.Image = co2_analysis.base

# Unify lists of segmentations (C5 is missing a few images).
for run in ["c1", "c2", "c3", "c4"]:
    segmentations[run] = segmentations[run][:68] + segmentations[run][72:]

# ! ---- Determine depth map through interpolation of measurements

# Need a darsia.Image with coordinate system. Fetch the first segmentation in C1.
# The data is not relevant
base_path = segmentations["c1"][0]
base = darsia.ScalarImage(
    np.load(base_path),
    width=2.8,
    height=1.5,
)
base_shape = base.img.shape[:2]
base_coordinate_system = base.coordinatesystem

# Interpolate data or fetch from cache
cache = Path("../cache/depth.npy")
if cache.exists():
    depth = np.load(cache)
else:
    x_meas = Path("../measurements/x_measurements.npy")
    y_meas = Path("../measurements/y_measurements.npy")
    d_meas = Path("../measurements/depth_measurements.npy")

    depth_measurements = (
        np.load(x_meas),
        np.load(y_meas),
        np.load(d_meas),
    )

    # Interpolate data
    depth = darsia.interpolate_measurements(
        depth_measurements, base_shape, base_coordinate_system
    )

    # Store in cache
    cache.parents[0].mkdir(parents=True, exist_ok=True)
    np.save(cache, depth)

# Define containers.
time = []
c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
spesial_comb = []
c2_c3_c4_overlap = []
other = []
total = []

# Define reference time for all runs:
time_ref = {}
for run in ["c1", "c2", "c3", "c4", "c5"]:
    time_ref[run] = read_time_from_path(segmentations[run][0])

# Do the comparison for separate time steps.
for time_index in range(len(segmentations["c1"])):

    # NOTE: Change if wanting to address different times
    if time_index % 10 != 0:
        continue

    rel_time = {}
    images = {}
    for run in ["c1", "c2", "c3", "c4", "c5"]:
        # Fetch segmentation
        images[run] = segmentations[run][time_index]

        # Determine (relative) time in minutes
        t_absolute = read_time_from_path(images[run])
        rel_time[run] = (t_absolute - time_ref[run]).total_seconds() / 60

    # Define name for plot by combining all times
    plot_name = (
        str(int(round(rel_time["c1"], 1)))
        + "_"
        + str(int(round(rel_time["c2"], 1)))
        + "_"
        + str(int(round(rel_time["c3"], 1)))
        + "_"
        + str(int(round(rel_time["c4"], 1)))
        + "_"
        + str(int(round(rel_time["c5"], 1)))
        + "_min.png"
    )
    plot_name = main_results / Path(plot_name)

    # Analyze the whole image for all runs and store to file
    ans, values, tot = whole_img(
        background,
        [images[run] for run in ["c1", "c2", "c3", "c4", "c5"]],
        depth,
        plot_name,
    )

    # Distribute the results.
    if ans:
        c1.append(values[0])
        c2.append(values[1])
        c3.append(values[2])
        c4.append(values[3])
        c5.append(values[4])
        spesial_comb.append(values[5])
        c2_c3_c4_overlap.append(values[6])
        other.append(values[7])
        total.append(tot)
        time.append(rel_time["c1"])
    else:
        print("NOTE: comparison was not successful.")

# Store results to excel.
df = pd.DataFrame()

df["Time [min]"] = time
df["C1"] = c1
df["C2"] = c2
df["C3"] = c3
df["C4"] = c4
df["C5"] = c5
df["spesial_comb"] = spesial_comb
df["c2_c3_c4_overlap"] = c2_c3_c4_overlap
df["other"] = other
df["total"] = total

df.to_excel(
    str(main_results / Path("fine_segmentation_whole_FL_weighted_colors.xlsx")),
    index=False,
)
