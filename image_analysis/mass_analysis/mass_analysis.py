from datetime import datetime, timedelta
from pathlib import Path

import cv2
import darsia
import numpy as np
import pandas as pd
import skimage
from scipy import interpolate
from skimage.measure import label, regionprops
from src.io import (concentration_to_csv, read_paths_from_user_data,
                    read_time_from_path, sg_to_csv, sw_to_csv)
from src.utils import interpolate_map

from injection_protocol import total_mass_co2_port1, total_mass_co2_port2

# ! ---- Provide path to the segmentations of C1, ..., C5.

seg_folders = {}
# Loop through all runs
for run in ["c1", "c2", "c3", "c4", "c5"]:
    # Read user-defined paths to images etc.
    _, _, _, results = read_paths_from_user_data("../data.json", run)
    seg_folders[run] = results / Path("npy_segmentation")

# ! ----  Injection times for C1, ..., C5.

inj_start_times = {
    "c1": datetime(2021, 11, 24, 8, 31, 0),
    "c2": datetime(2021, 12, 4, 10, 1, 0),
    "c3": datetime(2021, 12, 14, 11, 20, 0),
    "c4": datetime(2021, 12, 24, 9, 0, 0),
    "c5": datetime(2022, 1, 4, 11, 0, 0),
}

# ! ---- Determine depth map through interpolation of measurements

# Need a darsia.Image with coordinate system. Fetch the first segmentation in C1.
# The data is not relevant
base_path = list(sorted(seg_folders["c1"].glob("*.npy")))[0]
base = darsia.Image(
    np.load(base_path),
    width=2.8,
    height=1.5,
    color_space="GRAY",
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
    depth = interpolate_map(depth_measurements, base_shape, base_coordinate_system)

    # Store in cache
    cache.parents[0].mkdir(parents=True, exist_ok=True)
    np.save(cache, depth)

# ! ---- Build volumes

porosity = 0.44
Ny, Nx = base_shape
dx = 2.8 / Nx
dy = 1.5 / Ny
volume = np.multiply(porosity, depth) * dx * dy

# ! ---- Residual saturation
swi = np.zeros(base_shape, dtype=float)

# ! ---- Segmentation of the geometry providing a mask for the lower ESF layer.

# Reshape if needed
labels = cv2.resize(
    np.load(Path("../cache/labels_fine.npy")),
    tuple(reversed(base_shape)),
    interpolation=cv2.INTER_NEAREST,
)

# ! ---- Material properties.
dissolution_limit = 1.8  # kg / m**3

# ! ---- Analyze each run separately.

for i, directory in enumerate(seg_folders.values()):

    run_id = f"c{i+1}"
    print(f"Start with run {run_id}.")

    # Read user-defined paths to images etc.
    _, _, _, results = read_paths_from_user_data("../data.json", run_id)

    # ! ---- Measurements.

    # Fetch injection start
    inj_start = inj_start_times[run_id]

    # Fetch temporal pressure data
    pressure_data = Path(
        "../measurements/Florida_2021-11-24_2022-02-02_1669032742.xlsx"
    )
    df = pd.read_excel(pressure_data)

    # Extract time and add to the pressure data.
    date = [
        datetime.strptime(df.Dato.loc[i] + " " + df.Tid.loc[i], "%Y-%m-%d %H:%M")
        for i in range(len(df))
    ]
    df["Date"] = date

    # Reduce to the relevant times.
    df = df[
        (df.Date >= inj_start - timedelta(minutes=10))
        & (df.Date <= inj_start + timedelta(days=5) + timedelta(minutes=10))
    ]

    # Add time increments in minutes
    df["dt"] = ((df.Date - inj_start).dt.total_seconds()) / 60

    # Extract pressure data
    df["Lufttrykk"] = df.Lufttrykk + 3.125  # adjust for the height difference

    # Interpolate to get a function for atmospheric pressure over time (and scale pressures to bar)
    # pressure = interpolate.interp1d(time_pressure, 0.001 * atmospheric_pressures)
    pressure = interpolate.interp1d(df.dt.values, 0.001 * df.Lufttrykk.values)

    # ! ---- Data structures.

    # Choose a subregion
    subregions = {}
    subregions["all"] = None  # represents entire domain.
    subregions["boxA"] = np.array([[1.1, 0.6], [2.8, 0.0]])
    subregions["boxB"] = np.array([[0, 1.2], [1.1, 0.6]])

    # Create empty lists for plotting purposes
    time_vec = []
    total_mass_co2_vec = []
    total_mass_mobile_co2_vec = {}
    total_mass_dissolved_co2_vec = {}
    density_dissolved_co2_vec = {}
    for item in ["port1", "port2", "total"]:
        total_mass_mobile_co2_vec[item] = {}
        total_mass_dissolved_co2_vec[item] = {}
        for roi in ["boxA", "boxB", "esf", "all"]:
            total_mass_mobile_co2_vec[item][roi] = []
            total_mass_dissolved_co2_vec[item][roi] = []
        density_dissolved_co2_vec[item] = []

    # ! ---- Actual analysis for the specific run.

    # Loop through directory of segmentations (extract last images in which the plumes from the two injectors merge)
    seg_images = list(sorted(Path(directory).glob("*.npy")))
    for c, im in enumerate(seg_images):

        # Define reference time
        if c == 0:
            t_ref = read_time_from_path(im)

        # ! ---- Fetch data (time and segmentation)

        # Determine (relative) time in minutes
        t_absolute = read_time_from_path(im)
        t = (t_absolute - t_ref).total_seconds() / 60
        time_vec.append(t)

        # Fetch segmentation
        seg = np.load(im)

        # Fetch masks
        water_mask = seg == 0
        aq_mask = seg == 1
        gas_mask = seg == 2

        # ! ---- Determine saturations

        # Full water saturation aside of residual saturations in gas regions
        sw = np.ones((Ny, Nx), dtype=float)
        sw[seg == 2] = swi[seg == 2]

        # Complementary condition for gas saturation
        sg = 1 - sw

        # Saturation times vol
        sw_vol = {}
        sg_vol = {}
        sw_vol["total"] = np.multiply(sw, volume)
        sg_vol["total"] = np.multiply(sg, volume)

        # ! ---- Density map of free co2
        def external_pressure_to_density_co2(external_pressure: float) -> np.ndarray:
            """
            A conversion from pressure (in bar) to density of CO2 (in g/m^3)
            is given by the linear formula here. The formula is found by using
            linear regression on table values from NIST for standard conversion
            between pressure and density of CO2 at 23 Celsius. The extra added
            pressure due depth in the water is also corrected for, using the
            formula 0.1atm per meter, which amounts to 0.101325bar per meter.

            Arguments:
                external_pressure (float): external pressure, for example atmospheric pressure.

            Returns:
                (np.ndarray): array associating each pixel with the CO2 density.
            """
            # Defines a height map associating each pixel with its physical
            # height (distance from top)
            height_map = np.linspace(0, 1.5, Ny)[:, None] * np.ones(base_shape)

            # Make hardcoded conversion (based on interpolation of NIST data)
            return 1000 * (
                1.805726990977443 * (external_pressure + 0.101325 * height_map)
                - 0.009218969932330845
            )

        co2_g_density = external_pressure_to_density_co2(pressure(t))

        # ! ---- Free CO2 mass density
        mobile_co2_mass_density = {}
        mobile_co2_mass_density["total"] = np.multiply(sg_vol["total"], co2_g_density)

        #########################################################################
        # ! ---- Decompose segmentation into regions corresponding to injection in port 1 and port 2.

        # Label the segmentation and determine the region properties.
        seg_label = label(seg)
        regions = regionprops(seg_label)

        # Define the top right and rest of the image
        top_right = np.zeros_like(seg, dtype=bool)
        rest = np.zeros_like(seg, dtype=bool)

        # Loop through regions and check if centroid is in top right - if so discard that region.
        for i in range(len(regions)):
            if (
                0 < regions[i].centroid[0] < 2670  # y coordinate
                and 3450 < regions[i].centroid[1] < Nx  # x coordinate
            ):
                top_right[seg_label == regions[i].label] = True
            else:
                rest[seg_label == regions[i].label] = True

        # Decompose segmentation into the injections of first and seconds well.
        decomposed_seg = {}
        decomposed_seg["port1"] = np.zeros_like(seg, dtype=seg.dtype)
        decomposed_seg["port1"][rest] = seg[rest]

        decomposed_seg["port2"] = np.zeros_like(seg, dtype=seg.dtype)
        decomposed_seg["port2"][top_right] = seg[top_right]

        decomposed_aq = {}
        decomposed_gas = {}
        for item in ["port1", "port2"]:
            decomposed_aq[item] = decomposed_seg[item] == 1
            decomposed_gas[item] = decomposed_seg[item] == 2

        # Restrict co2_mass_density to port1 and port2
        for item in ["port1", "port2"]:
            sg_vol[item] = np.multiply(sg_vol["total"], decomposed_gas[item])
            mobile_co2_mass_density[item] = np.multiply(sg_vol[item], co2_g_density)

        #########################################################################
        # ! ---- Brief sparse mass analysis.

        # Determine total mass of co2 (injected through port 1 and port 2)
        total_mass_co2 = {}
        total_mass_co2["port1"] = total_mass_co2_port1(t)
        total_mass_co2["port2"] = total_mass_co2_port2(t)
        total_mass_co2["total"] = sum(
            [total_mass_co2[item] for item in ["port1", "port2"]]
        )

        total_mass_co2_g = {}
        for item in ["port1", "port2", "total"]:
            total_mass_co2_g[item] = np.sum(mobile_co2_mass_density[item])

        total_mass_co2_aq = {}
        for item in ["port1", "port2", "total"]:
            total_mass_co2_aq[item] = total_mass_co2[item] - total_mass_co2_g[item]

        total_volume_co2_aq = {}
        for item in ["port1", "port2"]:
            sw_vol[item] = np.multiply(sw_vol["total"], decomposed_aq[item])
            total_volume_co2_aq[item] = np.sum(sw_vol[item])
        total_volume_co2_aq["total"] = np.sum(sw_vol["port1"] + sw_vol["port2"])

        effective_density_co2_aq = {}
        for item in ["port1", "port2", "total"]:
            effective_density_co2_aq[item] = (
                0
                if total_volume_co2_aq[item] < 1e-9
                else total_mass_co2_aq[item] / total_volume_co2_aq[item]
            )

        # Build dense CO2(aq) mass
        concentration_co2_aq = np.zeros((Ny, Nx), dtype=float)

        # Pick dissolution limit in the gaseous area.
        concentration_co2_aq[seg == 2] = dissolution_limit

        # Pick efective concentrations in two plumes (in kg / m**3)
        if c < len(seg_images) - 6:
            for item in ["port1", "port2"]:
                concentration_co2_aq[decomposed_aq[item]] = (
                    effective_density_co2_aq[item] / 1000
                )
        else:
            # Treat case when plumes merge separately.
            concentration_co2_aq[seg == 1] = effective_density_co2_aq["total"] / 1000

        # Build spatial mass density integrated over volumes
        dissolved_co2_mass_density = {}
        for item in ["port1", "port2", "total"]:
            dissolved_co2_mass_density[item] = (
                np.multiply(sw_vol[item], concentration_co2_aq) * 1000
            )  # g / m**3

        # Total mass density
        co2_total_mass_density = {}
        for item in ["port1", "port2", "total"]:
            co2_total_mass_density[item] = (
                dissolved_co2_mass_density[item] + mobile_co2_mass_density[item]
            )

        # ! ---- Make roi analysis
        esf_label = 3  # hardcoded
        esf = labels == esf_label

        # Define boxes of interest
        box_a = (slice(int((1.5 - 1.1) / 1.5 * Ny), Ny), slice(int(1.1 / 2.8 * Nx), Nx))
        box_b = (
            slice(int((1.5 - 1.2) / 1.5 * Ny), int((1.5 - 0.6) / 1.5 * Ny)),
            slice(0, int(1.1 / 2.8 * Nx)),
        )
        entire_domain = np.ones((Ny, Nx), dtype=bool)

        subregions = {"boxA": box_a, "boxB": box_b, "esf": esf, "all": entire_domain}

        # ! ---- Collect results.
        total_mass_co2_vec.append(np.sum(co2_total_mass_density["total"]))

        for item in ["port1", "port2", "total"]:
            for key, roi in subregions.items():
                total_mass_mobile_co2_vec[item][key].append(
                    np.sum(mobile_co2_mass_density[item][roi])
                )
                total_mass_dissolved_co2_vec[item][key].append(
                    np.sum(dissolved_co2_mass_density[item][roi])
                )

            density_dissolved_co2_vec[item].append(effective_density_co2_aq[item])

        #########################################################################
        # ! ---- Store to file
        stem = im.stem

        # ! ---- Store dense data to file

        # ! ---- Concentration

        # Store numpy arrays
        filename_npy = stem.replace("_segmentation", "_concentration") + ".npy"
        npy_concentration_folder = Path("concentration_npy")
        full_filename_npy = results / npy_concentration_folder / Path(filename_npy)
        full_filename_npy.parents[0].mkdir(parents=True, exist_ok=True)
        np.save(full_filename_npy, concentration_co2_aq)

        # Store jpg images
        filename_jpg = stem.replace("_segmentation", "_concentration") + ".jpg"
        jpg_concentration_folder = Path("concentration_jpg")
        full_filename_jpg = results / jpg_concentration_folder / Path(filename_jpg)
        full_filename_jpg.parents[0].mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(full_filename_jpg),
            skimage.img_as_ubyte(
                np.clip(concentration_co2_aq / dissolution_limit, 0, 1)
            ),
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )

        # ! ---- Sw

        # Store numpy arrays
        filename_npy = stem.replace("_segmentation", "_sw") + ".npy"
        npy_sw_folder = Path("sw_npy")
        full_filename_npy = results / npy_sw_folder / Path(filename_npy)
        full_filename_npy.parents[0].mkdir(parents=True, exist_ok=True)
        np.save(full_filename_npy, sw)

        # Store jpg images
        filename_jpg = stem.replace("_segmentation", "_sw") + ".jpg"
        jpg_sw_folder = Path("sw_jpg")
        full_filename_jpg = results / jpg_sw_folder / Path(filename_jpg)
        full_filename_jpg.parents[0].mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(full_filename_jpg),
            skimage.img_as_ubyte(sw),
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )

        # ! ---- Sg

        # Store numpy arrays
        filename_npy = stem.replace("_segmentation", "_sg") + ".npy"
        npy_sg_folder = Path("sg_npy")
        full_filename_npy = results / npy_sg_folder / Path(filename_npy)
        full_filename_npy.parents[0].mkdir(parents=True, exist_ok=True)
        np.save(full_filename_npy, sg)

        # Store jpg images
        filename_jpg = stem.replace("_segmentation", "_sg") + ".jpg"
        jpg_sg_folder = Path("sg_jpg")
        full_filename_jpg = results / jpg_sg_folder / Path(filename_jpg)
        full_filename_jpg.parents[0].mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(full_filename_jpg),
            skimage.img_as_ubyte(sg),
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )

        #########################################################################
        # ! ---- Construct coarse respresentations of the spatial fields

        # Conversion rate needed to conserve the total area
        scaling = 4260 * 7952 / 280 / 150

        # Coarse volume (porosity x depth) - conserve volume
        coarse_volume = (
            cv2.resize(volume, (280, 150), interpolation=cv2.INTER_AREA) * scaling
        )

        # Coarse (binary) saturation - Boolean resize (nearest) - not conservative
        co2_coarse = skimage.img_as_bool(
            cv2.resize(
                skimage.img_as_ubyte(
                    seg >= 1,
                ),
                (280, 150),
                interpolation=cv2.INTER_AREA,
            )
        )

        co2_gas_coarse = skimage.img_as_bool(
            cv2.resize(
                skimage.img_as_ubyte(
                    seg == 2,
                ),
                (280, 150),
                interpolation=cv2.INTER_AREA,
            )
        )

        coarse_seg = np.zeros((150, 280), dtype=np.uint8)
        coarse_seg[co2_coarse] += 1
        coarse_seg[co2_gas_coarse] += 1

        coarse_sg = coarse_seg == 2
        coarse_sw = 1 - coarse_sg

        # Conserve the total mass

        # Integrate volume and saturations.
        # Decompose into contributions of the two different ports.
        coarse_decomposed_aq = {}
        for item in ["port1", "port2"]:
            coarse_decomposed_aq[item] = skimage.img_as_bool(
                cv2.resize(
                    skimage.img_as_ubyte(decomposed_aq[item]),
                    (280, 150),
                    interpolation=cv2.INTER_AREA,
                )
            )

        coarse_sw_vol = {}
        coarse_sw_vol["total"] = np.multiply(coarse_volume, coarse_sw)
        for item in ["port1", "port2"]:
            coarse_sw_vol[item] = np.multiply(
                coarse_sw_vol["total"], coarse_decomposed_aq[item]
            )

        # Total volumes
        coarse_total_volume_co2_aq = {}
        coarse_total_volume_co2_aq["port1"] = np.sum(coarse_sw_vol["port1"])
        coarse_total_volume_co2_aq["port2"] = np.sum(coarse_sw_vol["port2"])
        coarse_total_volume_co2_aq["total"] = np.sum(
            coarse_sw_vol["port1"] + coarse_sw_vol["port2"]
        )

        # Effective density of the dissolved CO2
        coarse_effective_density_co2_aq = {}
        for item in ["port1", "port2", "total"]:
            coarse_effective_density_co2_aq[item] = (
                0
                if coarse_total_volume_co2_aq[item] < 1e-9
                else total_mass_co2_aq[item] / coarse_total_volume_co2_aq[item]
            )

        # Create coarse versions of mobile co2 mass density without loss of mass
        coarse_mobile_co2_mass_density = {}
        for item in ["port1", "port2", "total"]:
            coarse_mobile_co2_mass_density[item] = (
                cv2.resize(
                    mobile_co2_mass_density[item],
                    (280, 150),
                    interpolation=cv2.INTER_AREA,
                )
                * scaling
            )

        # Build dense CO2(aq) mass
        coarse_concentration_co2_aq = np.zeros((150, 280), dtype=float)

        # Pick dissolution limit in the gaseous area.
        coarse_concentration_co2_aq[coarse_seg == 2] = dissolution_limit

        # Pick efective concentrations in two plumes (in kg / m**3)
        if c < len(seg_images) - 6:
            for item in ["port1", "port2"]:
                coarse_concentration_co2_aq[coarse_decomposed_aq[item]] = (
                    coarse_effective_density_co2_aq[item] / 1000
                )
        else:
            # Treat case when plumes merge separately.
            coarse_concentration_co2_aq[coarse_seg == 1] = (
                coarse_effective_density_co2_aq["total"] / 1000
            )

        # Build spatial mass density
        coarse_dissolved_co2_mass_density = {}
        for item in ["port1", "port2", "total"]:
            coarse_dissolved_co2_mass_density[item] = (
                np.multiply(coarse_sw_vol[item], coarse_concentration_co2_aq) * 1000
            )  # in g therefore scale

        # Total mass density
        coarse_co2_total_mass_density = {}
        for item in ["port1", "port2", "total"]:
            coarse_co2_total_mass_density[item] = (
                coarse_dissolved_co2_mass_density[item]
                + coarse_mobile_co2_mass_density[item]
            )

        # Build mass from element wise products - currently not used anywhere
        integrated_mass_co2 = np.multiply(
            volume,
            np.multiply(sw, concentration_co2_aq) + np.multiply(sg, dissolution_limit),
        )

        coarse_integrated_mass_co2 = np.multiply(
            coarse_volume,
            np.multiply(coarse_sw, coarse_concentration_co2_aq)
            + np.multiply(coarse_sg, dissolution_limit),
        )

        # ! ---- Store concentrations as coarse csv files, corresponding to 1cm by 1cm cells.

        filename_csv = stem.replace("_segmentation", "_concentration") + ".csv"
        csv_concentration_folder = Path("concentration_csv")
        full_filename_csv = results / csv_concentration_folder / Path(filename_csv)
        full_filename_csv.parents[0].mkdir(parents=True, exist_ok=True)
        concentration_to_csv(
            full_filename_csv,
            coarse_concentration_co2_aq,
            im.name,
        )

        # ! ---- Store saturations as coarse csv files, corresponding to 1cm by 1cm cells.

        # ! ---- Sg

        filename_csv = stem.replace("_segmentation", "_sg") + ".csv"
        csv_sg_folder = Path("sg_csv")
        full_filename_csv = results / csv_sg_folder / Path(filename_csv)
        full_filename_csv.parents[0].mkdir(parents=True, exist_ok=True)
        sg_to_csv(
            full_filename_csv,
            coarse_sg,
            im.name,
        )

        # ! ---- Sw

        filename_csv = stem.replace("_segmentation", "_sw") + ".csv"
        csv_sw_folder = Path("sw_csv")
        full_filename_csv = results / csv_sw_folder / Path(filename_csv)
        full_filename_csv.parents[0].mkdir(parents=True, exist_ok=True)
        sw_to_csv(
            full_filename_csv,
            coarse_sw,
            im.name,
        )

    # ! ---- Collect all data in excel sheets

    for item in ["port1", "port2", "total"]:

        df = pd.DataFrame()
        df["Time_[min]"] = time_vec

        df["Total_CO2"] = total_mass_co2_vec

        df["Mobile_CO2_[g]"] = total_mass_mobile_co2_vec[item]["all"]
        df["Dissolved_CO2_[g]"] = total_mass_dissolved_co2_vec[item]["all"]

        for roi in ["boxA", "boxB", "esf"]:
            df[f"Mobile_CO2_{roi}_[g]"] = total_mass_mobile_co2_vec[item][roi]
            df[f"Dissolved_CO2_{roi}_[g]"] = total_mass_dissolved_co2_vec[item][roi]

        if item in ["port1", "port2"]:
            df[f"Concentration_CO2_{item}"] = density_dissolved_co2_vec[item]

        Path(results / Path("sparse_data")).mkdir(parents=True, exist_ok=True)
        excel_path = results / Path("sparse_data") / Path(f"mass_analysis_{item}.xlsx")
        df.to_excel(str(excel_path), index=None)
