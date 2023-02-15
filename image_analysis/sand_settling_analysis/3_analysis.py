"""
Determine intensity of sand settling over the course of the different runs.

"""

import darsia
import numpy as np

# ! ----- Choose which runs to compare

comparison = "c1_vs_c5"

# ! ----- Preliminaries - prepare two images for image registration

# C1
c1_npy = np.load("images/c1.npy")
c1_img = darsia.Image(c1_npy, width=2.8, height=1.5)
c1_water = np.load("labels/c1_water.npy")
c1_mask_img = darsia.Image(np.logical_not(c1_water), width=2.8, height=1.5)

# C2
c2_npy = np.load("images/c2.npy")
c2_img = darsia.Image(c2_npy, width=2.8, height=1.5)
c2_water = c1_water.copy()
c2_mask_img = darsia.Image(np.logical_not(c2_water), width=2.8, height=1.5)

# C3
c3_npy = np.load("images/c3.npy")
c3_img = darsia.Image(c3_npy, width=2.8, height=1.5)
c3_water = np.load("labels/c3_water.npy")
c3_mask_img = darsia.Image(np.logical_not(c3_water), width=2.8, height=1.5)

# C4
c4_npy = np.load("images/c4.npy")
c4_img = darsia.Image(c4_npy, width=2.8, height=1.5)
c4_water = c3_water.copy()
c4_mask_img = c3_mask_img.copy()

# C5
c5_npy = np.load("images/c5.npy")
c5_img = darsia.Image(c5_npy, width=2.8, height=1.5)
c5_water = c3_water.copy()
c5_mask_img = c3_mask_img.copy()

# Fix a reference image
if comparison == "c1_vs_c2":

    # Images
    img_ref = c1_img.copy()
    img_dst = c2_img.copy()

    # Masks
    mask_ref = c1_mask_img.copy()
    mask_dst = c2_mask_img.copy()

elif comparison == "c2_vs_c3":
    # Images
    img_ref = c2_img.copy()
    img_dst = c3_img.copy()

    # Masks
    mask_ref = c2_mask_img.copy()
    mask_dst = c3_mask_img.copy()

elif comparison == "c3_vs_c4":
    # Images
    img_ref = c3_img.copy()
    img_dst = c4_img.copy()

    # Masks
    mask_ref = c3_mask_img.copy()
    mask_dst = c4_mask_img.copy()

elif comparison == "c4_vs_c5":
    # Images
    img_ref = c4_img.copy()
    img_dst = c5_img.copy()

    # Masks
    mask_ref = c4_mask_img.copy()
    mask_dst = c5_mask_img.copy()

elif comparison == "c1_vs_c5":
    # Images
    img_ref = c1_img.copy()
    img_dst = c5_img.copy()

    # Masks
    mask_ref = c1_mask_img.copy()
    mask_dst = c5_mask_img.copy()


# ! ---- Multiscale image registration

# Define multilevel config
config_image_registration = {
    # Define hierarchy of patches
    "N_patches": [[32, 16], [200, 100]],
    # Define a relative overlap, this makes it often slightly
    # easier for the feature detection.
    # "rel_overlap": 0.1,
    "rel_overlap": 1.0,
    # Add some tuning parameters for the feature detection
    # (these are actually the default values and could be
    # also omitted.
    "max_features": 200,
    "tol": 0.05,
    "verbosity": 0,
}
image_registration = darsia.ImageRegistration(
    img_dst=img_dst, mask_dst=mask_dst, **config_image_registration
)

# Determine the effective deformation
transformed_ref = image_registration(img_ref, mask_ref)

# Plot the result
image_registration.plot(scaling=1.0, mask=mask_ref)
