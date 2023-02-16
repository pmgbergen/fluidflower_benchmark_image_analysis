#!/bin/bash

# Segment geometry and detect facies
cd geometry_segmentation
python geometry_segmentation_coarse.py
python geometry_segmentation_fine.py

# Run segmentation for all experimental runs C1-5
cd ../phase_segmentation
python phase_segmentation.py

# Extract the spatial concentration and saturation maps,
# total mass etc. for all runs, based on the phase
# segmentation and controlled injection protocol.
# The outcome provides means to determine qty 2, 3, 4, 6
# in the sparse data analysis.
cd ../mass_analysis
python concentration_analysis.py

# Run mixing analysis, required to determine qty 5
# in the sparse data analysis.
cd ../mixing_analysis
python mixing_analysis.py
python mixing_analysis_post.py

# Manual visit of the following files optional
# providing additional results displayed in the
# final analysis:
# cd ..

# Generate figures as in SM5 to study physical variability.
# python phase_segmentation_comparison/phase_segmentation_comparison.py

# Count fingers and the length of the phase segmentation
# python finger_analysis/finger_analysis.py
# cd finger_analysis
# ./run_finger_analysis.sh

# Quantify the sand settling between the runs.
# cd ../sand_settling
# ./run.sh
