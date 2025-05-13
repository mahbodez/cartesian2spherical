from spherical.transforms import (
    nifti_to_spherical,
    spherical_to_nifti,
)


# ------------------------------------------------------------------------------
# Demo — synthetic spherical pattern
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- NIfTI file transform demo ---
    nifti_in_path = "./mri_14725586.nii.gz"
    nifti_sph_path = "./output_spherical.nii"
    nifti_meta_path = "./output_meta.json"
    nifti_to_spherical(
        nifti_in_path,
        nifti_sph_path,
        out_meta_path=nifti_meta_path,
        r_bins=256,
        theta_bins=256,
        phi_bins=256,
        center="com",
        interp_order=5,
        equal_area_phi=True,
        pad_mode="constant",
        cval="min",
    )
    spherical_to_nifti(
        nifti_sph_path,
        "./output_cartesian.nii",
        grid_meta_path=nifti_meta_path,
        interp_order=5,
        pad_mode="constant",
        cval="min",
    )
