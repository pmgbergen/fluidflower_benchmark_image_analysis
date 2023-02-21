"""
Phase segmentation for all runs of the FluidFlower International Benchmark.
"""
from src.io import read_paths_from_user_data
from src.largerigco2analysis import LargeRigCO2Analysis

# Fix config file for all runs.
config = "./config.json"

# Loop through all runs
for run in ["c1", "c2", "c3", "c4", "c5"]:

    # Read user-defined paths to images etc.
    images, baseline, _, results = read_paths_from_user_data(
        "../data.json",
        run
    )

    # Define FluidFlower based on a full set of basline images
    co2_analysis = LargeRigCO2Analysis(
        baseline=baseline,  # paths to baseline images
        config=config,  # path to config file
        results=results,  # path to results directory
    )

    # Perform standardized CO2 segmentation on all images
    co2_analysis.batch_analysis(images, plot_contours=False)
