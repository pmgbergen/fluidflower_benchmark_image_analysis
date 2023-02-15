"""
Post analysis script analyzing the evolution of fingers in runs C1-5.
"""
from pathlib import Path

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
import skimage
from src.io import read_paths_from_user_data, read_time_from_path
from src.largefluidflower import LargeFluidFlower

# Run analysis
for run in ["c1", "c2", "c3", "c4", "c5"]:

    # Read user-defined paths to images etc.
    images, _, _, results = read_paths_from_user_data("../data.json", run)

    # Fetch reference time
    ref_time = read_time_from_path(images[0])

    # Setup up fluidflower
    base = images[0]
    config = Path("../phase_segmentation/config.json")
    fluidflower = LargeFluidFlower(base, config)

    # Fetch segmentations.
    segmentations = list(sorted((results / Path("npy_segmentation")).glob("*.npy")))
    num_segmentations = len(segmentations)

    # Focus on small portion - outside this time interval, fingering either
    # does not take place or has already exited box c.
    start_index = 20
    end_index = 80
    segmentations = list(sorted(segmentations))[start_index:end_index]
    images = images[start_index:end_index]

    # Contour analysis - initialize new for each run.
    contour_analysis = darsia.ContourAnalysis(verbosity=False)
    co2_g_analysis = darsia.ContourAnalysis(verbosity=False)
    contour_evolution_analysis = darsia.ContourEvolutionAnalysis()

    # Keep track of number, length of fingers
    total_num_fingers = []
    length_fingers = []
    height_co2g = []
    rel_time = []

    # Start with a fixed one - as Darsia Image
    for i in range(num_segmentations):

        # Convert segmentation to Image
        segmentation = darsia.Image(np.load(segmentations[i]), width=2.8, height=1.5)
        fluidflower.load_and_process_image(images[i])
        original = fluidflower.img

        # Add timestamp from title
        time = read_time_from_path(images[i])
        segmentation.timestamp = time
        original.timestamp = time
        relative_time_hours = (time - ref_time).total_seconds() / 3600
        rel_time.append(relative_time_hours)

        # Plot the segmentation and box C - set to true for active display
        if False:
            segmentation_img = np.zeros((*segmentation.img.shape[:2], 3), dtype=float)
            for i in range(3):
                segmentation_img[:, :, i] = segmentation.img / np.max(segmentation.img)
            segmentation_img = skimage.img_as_ubyte(segmentation_img)

            # Add box A
            contours_box_A, _ = cv2.findContours(
                skimage.img_as_ubyte(fluidflower.mask_box_A),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(segmentation_img, contours_box_A, -1, (180, 180, 180), 3)

            # Add box C
            contours_box_C, _ = cv2.findContours(
                skimage.img_as_ubyte(fluidflower.mask_box_C),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(segmentation_img, contours_box_C, -1, (180, 180, 180), 3)

            plt.imshow(segmentation_img)
            plt.show()

        # Apply contour analysis for box C - only (all) CO2.
        contour_analysis.load_labels(
            img=segmentation,
            roi=fluidflower.box_C,
            values_of_interest=[1, 2],
        )

        # Number of fingers of total CO2
        num_fingers = contour_analysis.number_peaks()
        total_num_fingers.append(num_fingers)
        print(f"Number of fingers: {num_fingers}")

        # Length of interface between CO2 and water, without the box.
        single_length_fingers = contour_analysis.length()
        length_fingers.append(single_length_fingers)
        print(f"Contour length: {single_length_fingers}")

        # Contour tip analysis.
        tips, valleys = contour_analysis.fingers()

        # Build up evolution of contours
        contour_evolution_analysis.add(tips, valleys)

        # Plot finger tips onto image - set to true for active display
        if False and num_fingers > 17:
            plt.figure("Original image with finger tips")
            plt.imshow(original.img)
            plt.scatter(
                tips[:, 0, 0] + fluidflower.box_C_roi[0, 0],
                tips[:, 0, 1] + fluidflower.box_C_roi[0, 0],
                c="r",
                s=1,
            )
            # Uncomment for storing to file
            # plt.savefig("tips.svg", format="svg", dpi=1000)
            plt.show()

    # Collect data
    arr_num_fingers = np.transpose(
        np.vstack((np.array(rel_time), np.array(total_num_fingers)))
    )

    arr_finger_length = np.transpose(
        np.vstack((np.array(rel_time), np.array(length_fingers)))
    )

    # Store number data to file
    (results / Path("sparse_data")).mkdir(parents=True, exist_ok=True)
    finger_header = f"Time in hours, number of finger tips in box C"
    fmt = "%f", "%d"
    np.savetxt(
        results / Path("sparse_data") / Path("number_fingers.csv"),
        arr_num_fingers,
        fmt=fmt,
        delimiter=",",
        header=finger_header,
    )
    finger_header = f"Time in hours, finger length in box C"
    fmt = "%f", "%f"
    np.savetxt(
        results / Path("sparse_data") / Path("length_fingers.csv"),
        arr_finger_length,
        fmt=fmt,
        delimiter=",",
        header=finger_header,
    )
