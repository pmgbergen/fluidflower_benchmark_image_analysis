"""
Module collecting I/O utilities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np


def read_time_from_path(path: Union[Path, str]) -> datetime:
    """
    Method to read hardcoded datetime path to datetime format.

    The conversion expects a very specific style for file name.
    They need to start with the date in yyMMdd format, following
    by an underscore, the keyword 'time' and then the time of the
    day in the format HHmmss. An example name is for instance:

    "folder/subfolder/211124_time083942_DSC00139.JPG"

    This method returns the datetime corresponding to 24. Nov 2021,
    8 hours, 39 minutes, 42 seconds.

    Args:
        path (Path or str): path to file; both full path and path without
            folders are supported.

    Returns:
        datetime.datetime: time
    """
    # Convert to Path
    if isinstance(path, str):
        path = Path(path)

    # Clean path from all folders
    path_file = path.name

    # Identify the first 6 letters as date in the format yyMMdd
    date: str = path_file[:6]

    # Identify letters 12-17 (starting to count from 1) as time in the format HHmmss
    time: str = path_file[11:17]

    # Convert to datetime
    return datetime.strptime(date + " " + time, "%y%m%d %H%M%S")


def read_paths_from_user_data(
    path: Union[Path, str],
    key: Optional[str] = None,
) -> tuple[list[Path], list[Path], Path, Path]:
    """
    Method to read paths to images, baseline images and conig from
    standardized input config file.

    Args:
        path (str or Path): path to user-specific json file.
        key (str, optional): key for nested json files.

    Returns:
        list of Path: paths to images
        list of Path: paths to baseline images
        Path: path to config file
        Path: path to results directory
    """
    # Convert to Path
    if isinstance(path, str):
        path = Path(path)

    # Fetch json file
    with open(path, "r") as openfile:
        user_data = json.load(openfile)

    # If key provided fetch subdirectory.
    if key is not None:
        user_data = user_data[key]

    # Define the location for images of C1
    images_folder = Path(user_data["images"])
    file_ending = user_data["file ending"]
    all_images = list(sorted(images_folder.glob(file_ending)))

    # Extract basline images and actual images of the injection
    num_baseline_images = user_data["number baseline images"]
    baseline = all_images[:num_baseline_images]
    images = all_images[num_baseline_images:]

    # Path to analysis specific config file
    config = Path(user_data["config"])

    # Path to results directory, create if not existent yet
    results = Path(user_data["results"])
    results.mkdir(parents=True, exist_ok=True)

    return images, baseline, config, results


def segmentation_to_csv(path: Path, arr: np.ndarray, inFileName: Path) -> None:
    """Store numpy array of segmentation (water, CO2(w), CO2(g)) as csv file.

    Args:
        path (Path): path for dst
        array (np.ndarray): array to be stored
        inFileName (Path): origin image

    """
    # Build csv file with standard text on top.
    header = f"Generated from {str(inFileName)}."

    header += (
        "\nEach entry identifies the phase distribution in a 1 cm by 1 cm cell, where"
    )

    header += "\n0 = pure water, 1 = water with dissolved CO2, 2 = gaseous CO2."

    header += "\n2d array representing the laser grid of the benchmark description,"

    header += (
        "\nordered row-wise from top to bottom, and column-wise from left to right."
    )

    header += "\nThe first entry corresponds to the cell centered at coordinates (0.035 m, 1.525 m)"

    header += "\nw.r.t. the spatial maps outlined in Section 3.1 of the description."

    # Store array with preceding header
    np.savetxt(path, arr, fmt="%d", delimiter=",", header=header)


def concentration_to_csv(path: Path, arr: np.ndarray, inFileName: Path) -> None:
    """Store numpy of CO2 concentration as csv file.

    Args:
        path (Path): path for dst
        array (np.ndarray): array to be stored
        inFileName (Path): origin image

    """
    # Build csv file with standard text on top.
    header = f"Generated from {str(inFileName)}."

    header += "\nEach entry identifies the CO2 mass concentration in the water phase in a 1 cm by 1 cm cell."

    header += "\n2d array representing the laser grid of the benchmark description,"

    header += (
        "\nordered row-wise from top to bottom, and column-wise from left to right."
    )

    header += "\nThe first entry corresponds to the cell centered at coordinates (0.035 m, 1.525 m)"

    header += "\nw.r.t. the spatial maps outlined in Section 3.1 of the description."

    # Store array with preceding header
    np.savetxt(path, arr, fmt="%.4f", delimiter=",", header=header)


def sw_to_csv(path: Path, arr: np.ndarray, inFileName: Path) -> None:
    """Store numpy array of water saturation of as csv file.

    Args:
        path (Path): path for dst
        array (np.ndarray): array to be stored
        inFileName (Path): origin image

    """
    # Build csv file with standard text on top.
    header = f"Generated from {str(inFileName)}."

    header += "\nEach entry identifies the water saturation in a 1 cm by 1 cm cell."

    header += "\n2d array representing the laser grid of the benchmark description,"

    header += (
        "\nordered row-wise from top to bottom, and column-wise from left to right."
    )

    header += "\nThe first entry corresponds to the cell centered at coordinates (0.035 m, 1.525 m)"

    header += "\nw.r.t. the spatial maps outlined in Section 3.1 of the description."

    # Store array with preceding header
    np.savetxt(path, arr, fmt="%.4f", delimiter=",", header=header)


def sg_to_csv(path: Path, arr: np.ndarray, inFileName: Path) -> None:
    """Store numpy of gas saturation as csv file.

    Args:
        path (Path): path for dst
        array (np.ndarray): array to be stored
        inFileName (Path): origin image

    """
    # Build csv file with standard text on top.
    header = f"Generated from {str(inFileName)}."

    header += "\nEach entry identifies the gas saturation in a 1 cm by 1 cm cell."

    header += "\n2d array representing the laser grid of the benchmark description,"

    header += (
        "\nordered row-wise from top to bottom, and column-wise from left to right."
    )

    header += "\nThe first entry corresponds to the cell centered at coordinates (0.035 m, 1.525 m)"

    header += "\nw.r.t. the spatial maps outlined in Section 3.1 of the description."

    # Store array with preceding header
    np.savetxt(path, arr, fmt="%.4f", delimiter=",", header=header)
