"""
Preparations for the analysis of FluidFlower benchmark.
Provide segmentation of the geometry. Aim for detecting
'many' facies, and do not prioritize precision.
"""
from src.io import read_paths_from_user_data
from src.largerigco2analysis import LargeRigCO2Analysis

# Read user-defined paths to images, number of baseline images,
# and config file. Use C1 here, any of the images works. Yet the
# images for C1 are cleanest.
images, baseline, _, results = read_paths_from_user_data("../data.json", "c1")

# Specify the config file for segmenting the geometry.
config = "./config_geometry_segmentation_fine.json"

# Define FluidFlower analysis - implicitly generates a
# labeled image of the domain.
_ = LargeRigCO2Analysis(
    baseline=baseline,  # paths to baseline images
    config=config,  # path to config file
    results=results,  # path to results directory
    update_setup=True,  # strictly request update of cache
)
