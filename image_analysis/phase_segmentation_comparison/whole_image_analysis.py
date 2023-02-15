from pathlib import Path

import cv2
import darsia
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")


def whole_img(
    baseline: darsia.Image,
    segmentation_path: list[Path],
    depth_map: np.ndarray,
    plot_name: Path,
):
    """
    Segmentation comparison for the entire domain.

    Args:
        baseline (darsia.Image): baseline image
        segmentation_path (list of Path): paths to segmentations to be compared.
        depth_map (np.ndarray): depth map
        plot_name (Path): path for storing the comparison

    Returns:
        bool: success flag
        list: weighted colors
        list: total colors
    """
    # Create segmentation object (It is necessary to provide the number of segmentations that you want to compare)
    # If no colors are provided, default colors will be chosen. NOTE: Number of colors must match the number of segmentated images.
    segmentationComparison = darsia.SegmentationComparison()

    # Tailored for the benchmark and requiring 5 runs
    assert len(segmentation_path) == 5

    # Create the comparison array (Here as many segmentations as desirable can be provided)
    comparison = segmentationComparison.compare_segmentations_binary_array(
        np.load(segmentation_path[0]),
        np.load(segmentation_path[1]),
        np.load(segmentation_path[2]),
        np.load(segmentation_path[3]),
        np.load(segmentation_path[4]),
    )

    # Create color palette
    sns_palette = np.array(sns.color_palette())
    gray = np.array([[0.3, 0.3, 0.3], [1, 1, 1]])
    palette = np.append(sns_palette[:6], gray, axis=0)

    # Image to be filled with colors depending on segmentation overlap
    colored_comparison = np.zeros(
        comparison.shape[:2] + (3,),
        dtype=np.uint8,
    )

    # List of unique combinations
    unique_combinations = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]

    # List of combinations including c2, c3, and c4 (the 1, 2, 3 is for the relative
    # position in the presence lists, i.e., we will get all combinations of the form [x,1,1,1,x],
    # it is of course only 4 different ones in this specific case, but this is a general function)
    comb_c2_c3_c4 = segmentationComparison.get_combinations(
        1, 2, 3, num_segmentations=5
    )

    # Get bolean array of true-values for all combinations of c2, c3, and c4.
    c_2_c3_c4_bool = np.zeros(comparison.shape[:2], dtype=bool)
    for combination in comb_c2_c3_c4:
        c_2_c3_c4_bool += np.all(comparison == combination, axis=2)

    # -------------------------------------#
    # Get bolean array of true-values for all combinations between two runs for c2-c3-c4.
    spes_bool = np.zeros(comparison.shape[:2], dtype=bool)

    comb_c2_c3 = segmentationComparison.get_combinations(1, 2, num_segmentations=5)
    for combination in comb_c2_c3:
        spes_bool += np.all(comparison == combination, axis=2)

    comb_c3_c4 = segmentationComparison.get_combinations(2, 3, num_segmentations=5)

    for combination in comb_c3_c4:
        spes_bool += np.all(comparison == combination, axis=2)

    comb_c2_c4 = segmentationComparison.get_combinations(1, 3, num_segmentations=5)

    for combination in comb_c2_c4:
        spes_bool += np.all(comparison == combination, axis=2)
    # -------------------------------------#

    # Fill the colored image. Start by coloring all pixels that have any segmentation present at all with the light gray color
    colored_comparison[np.any(comparison != [0, 0, 0, 0, 0], axis=2)] = (
        palette[7] * 255
    ).astype(np.uint8)

    # Fill the colored image with the spesial combinations
    colored_comparison[spes_bool] = (palette[5] * 255).astype(np.uint8)

    # Color the unique combinations
    for c, i in enumerate(unique_combinations):
        colored_comparison[np.all(comparison == i, axis=2)] = (palette[c] * 255).astype(
            np.uint8
        )

    # Color the combination where c2, c3 and c4 are included
    colored_comparison[c_2_c3_c4_bool] = (palette[6] * 255).astype(np.uint8)

    # NOTE: If several of these computations are to be done on images of the same size, save the depth_map and feed it to the color_fractions() method instead of depth_measurements. That will result in a HUGE time save.
    (
        weighted_colors,
        color_fractions,
        colors,
        total_color,
        depth_map,
    ) = segmentationComparison.color_fractions(
        colored_comparison,
        colors=(palette * 255).astype(np.uint8),
        depth_map=depth_map,
    )

    # Create plot and store to file
    figure, axes = plt.subplots(figsize=(20, 10))

    # create legend paches
    labels = ["C1", "C2", "C3", "C4", "C5", "spesial_comb", "C2+C3+C4", "Other"]

    for i in range(len(labels)):
        labels[i] = labels[i] + " " + str(round(weighted_colors[i], 2))
    patch = []
    for i in range(8):
        patch.append(mpatches.Patch(color=palette[i], label=labels[i]))

    # Process the comparison image
    processed_comparison_image = segmentationComparison._post_process_image(
        colored_comparison, unique_colors=colors, opacity=0.6, contour_thickness=10
    )

    plt.imshow(cv2.resize(baseline.img, tuple(reversed(colored_comparison.shape[:2]))))
    plt.imshow(processed_comparison_image)
    plt.legend(handles=patch, bbox_to_anchor=(0.85, 1), loc=2, borderaxespad=0.0)

    boxes = [
        np.array([[1.1, 0.6], [2.8, 0.0]]),  # boxA
        np.array([[0.0, 1.2], [1.1, 0.0]]),  # boxB*
        np.array([[1.1, 1.2], [2.8, 0.6]]),
    ]  # boxD

    colors = [[0.2, 0.8, 0.2], [1, 1, 1], [1, 1, 0]]
    plt.axis("off")
    gs = plt.tight_layout()
    plt.savefig(plot_name, dpi=100)

    try:
        return True, weighted_colors, total_color
    except:
        False, None
