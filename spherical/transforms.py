import numpy as np
from scipy.ndimage import map_coordinates
from .nifti_utils import load_nifti_file, save_nifti_file
import json
from scipy.ndimage import center_of_mass


# ------------------------------------------------------------------------------
# Utility: equal-area latitude grid  (φ bins have identical surface area on S²)
# ------------------------------------------------------------------------------
def equal_area_phi_bins(n_phi: int):
    """
    Return array of φ centres (radians) so that each φ slice covers equal area
    on the unit sphere.
    """
    u = (np.arange(n_phi) + 0.5) / n_phi  # uniform in [0,1]
    return np.arccos(1 - 2 * u)  # φ centre for each band


# ------------------------------------------------------------------------------
# Forward transform: Cartesian volume → spherical tensor
# ------------------------------------------------------------------------------
def cartesian_to_spherical(
    vol: np.ndarray,
    r_bins: int = 128,
    theta_bins: int = 256,
    phi_bins: int = 128,
    center: tuple[float, float, float] | None = None,
    interp_order: int = 3,
    equal_area_phi: bool = True,
    pad_mode: str = "constant",
    cval: float | str = 0.0,
):
    """
    Parameters
    ----------
    vol : (D,H,W) ndarray
        Input MRI volume in Cartesian voxel space.
    r_bins, theta_bins, phi_bins : int
        Resolution of spherical grid.
    center : (cz, cy, cx) or None
        Origin in voxel coordinates. If None, geometric centre-of-mass is used.
    interp_order : int
        Spline order for `map_coordinates` (0=NN … 5 quintic). Cubic (3) is a
        good trade-off between accuracy and ringing.
    equal_area_phi : bool
        If True, latitude bins are equal-area; otherwise they are linearly
        spaced in φ.
    pad_mode, cval : str, float
        Boundary handling for `map_coordinates`.
    Returns
    -------
    sph : (R, Φ, Θ) ndarray
    grid_meta : dict
        Metadata needed for the inverse mapping (bin edges, origin, etc.).
    """
    if vol.ndim != 3:
        raise ValueError("Expected (D,H,W) array.")

    D, H, W = vol.shape
    if center is None:
        cz, cy, cx = ((D - 1) / 2.0, (H - 1) / 2.0, (W - 1) / 2.0)
    else:
        cz, cy, cx = center

    if isinstance(cval, str):
        if cval == "mean":
            cval = np.mean(vol)
        elif cval == "median":
            cval = np.median(vol)
        elif cval == "min":
            cval = np.min(vol)
        elif cval == "max":
            cval = np.max(vol)
        else:
            raise ValueError(f"Unknown cval: {cval}")

    # --- replace the old “inside‐cube” radius with a corner‐to‐center radius ---
    # compute distance to each of the 8 corners, then take the maximum
    corner_offsets = [
        (zc, yc, xc) for zc in (0, D - 1) for yc in (0, H - 1) for xc in (0, W - 1)
    ]
    r_max = max(
        np.linalg.norm(np.array([cz - zc, cy - yc, cx - xc]))
        for zc, yc, xc in corner_offsets
    )

    # radial bin centres (include the origin at r=0)
    r_centres = np.linspace(0.0, r_max, r_bins, dtype=np.float32)

    # θ and φ bin centres
    theta_centres = (np.arange(theta_bins) + 0.5) * (2 * np.pi / theta_bins)

    if equal_area_phi:
        phi_centres = equal_area_phi_bins(phi_bins).astype(np.float32)
    else:
        phi_centres = (np.arange(phi_bins) + 0.5) * (np.pi / phi_bins)

    # meshgrid in (r, φ, θ) order
    r, phi, theta = np.meshgrid(r_centres, phi_centres, theta_centres, indexing="ij")

    # spherical → Cartesian (floating-point voxel coords)
    x = r * np.sin(phi) * np.cos(theta) + cx
    y = r * np.sin(phi) * np.sin(theta) + cy
    z = r * np.cos(phi) + cz

    coords = np.vstack([z.ravel(), y.ravel(), x.ravel()])  # order: z,y,x
    sph = map_coordinates(
        vol,
        coords,
        order=interp_order,
        mode=pad_mode,
        cval=cval,
    ).reshape(r_bins, phi_bins, theta_bins)

    grid_meta = dict(
        center=np.array([cz, cy, cx]),
        r_max=r_max,
        r_bins=r_bins,
        theta_bins=theta_bins,
        phi_bins=phi_bins,
        equal_area_phi=equal_area_phi,
        orig_shape=vol.shape,  # <-- add this
    )
    return sph.astype(vol.dtype, copy=False), grid_meta


# ------------------------------------------------------------------------------
# Inverse transform: spherical tensor → Cartesian volume
# ------------------------------------------------------------------------------
def spherical_to_cartesian(
    sph: np.ndarray,
    cart_shape: tuple[int, int, int],
    grid_meta: dict,
    interp_order: int = 3,
    pad_mode: str = "constant",
    cval: float | str = 0.0,
    **kwargs,
):
    """
    Reconstruct Cartesian volume from spherical tensor.

    Parameters
    ----------
    sph : (R, Φ, Θ) ndarray
        Spherical volume.
    cart_shape : (D,H,W)
        Desired Cartesian grid.
    grid_meta : dict
        Returned by `vol_to_spherical`, supplies bin definitions.
    """
    R, F, T = sph.shape
    D, H, W = cart_shape
    cz, cy, cx = grid_meta["center"]
    r_max = grid_meta["r_max"]
    equal_area_phi = grid_meta["equal_area_phi"]

    if isinstance(cval, str):
        if cval == "mean":
            cval = np.mean(sph)
        elif cval == "median":
            cval = np.median(sph)
        elif cval == "min":
            cval = np.min(sph)
        elif cval == "max":
            cval = np.max(sph)
        else:
            raise ValueError(f"Unknown cval: {cval}")

    # Bin edges are as in forward mapping
    r_centres = np.linspace(0.0, r_max, R, dtype=np.float32)
    theta_centres = (np.arange(T) + 0.5) * (2 * np.pi / T)

    if equal_area_phi:
        phi_centres = equal_area_phi_bins(F).astype(np.float32)
    else:
        phi_centres = (np.arange(F) + 0.5) * (np.pi / F)

    # Precompute steps for index mapping
    dr = r_centres[1] - r_centres[0] if R > 1 else 1.0
    dtheta = 2 * np.pi / T
    # For equal-area φ the mapping is non-linear; interpolate via monotonic array
    if equal_area_phi:
        # Numerical inverse: φ → fractional index
        def phi_to_idx(phi_val):
            return np.interp(phi_val, phi_centres, np.arange(F))

    else:
        dphi = np.pi / F

    # Cartesian grid
    z, y, x = np.indices(cart_shape)
    dx, dy, dz = x - cx, y - cy, z - cz

    r = np.sqrt(dx**2 + dy**2 + dz**2)
    theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
    phi = np.arccos(np.clip(dz / np.maximum(r, 1e-6), -1, 1))

    # keep a mask of true-center voxels
    flat_r = r.ravel()
    center_mask = flat_r == 0

    # Map to fractional indices  (clip φ,r to avoid pole/corner artifacts, wrap θ)
    # radial → [0,R-1]
    r_idx = np.clip(r / dr, 0, R - 1)
    # longitude → [0,T)
    theta_idx = (theta / dtheta) % T
    # latitude → fractional bin, then clip to [0,F-1]
    if equal_area_phi:
        phi_idx = phi_to_idx(phi)
    else:
        phi_idx = phi / dphi
    phi_idx = np.clip(phi_idx, 0, F - 1)

    coords = np.vstack([r_idx.ravel(), phi_idx.ravel(), theta_idx.ravel()])

    # main interpolation
    recon = map_coordinates(
        sph,
        coords,
        order=interp_order,
        mode=pad_mode,
        cval=cval,
    ).reshape(cart_shape)

    # fix the exact center voxel(s) by averaging over all φ,θ at r=0
    if center_mask.any():
        center_value = sph[0, :, :].mean()
        recon_flat = recon.ravel()
        recon_flat[center_mask] = center_value
        recon = recon_flat.reshape(cart_shape)

    return recon.astype(sph.dtype, copy=False)


def nifti_to_spherical(
    in_path: str,
    out_sph_path: str,
    out_meta_path: str | None = None,
    r_bins: int = 128,
    theta_bins: int = 256,
    phi_bins: int = 128,
    center: tuple[float, float, float] | str | None = None,
    interp_order: int = 3,
    equal_area_phi: bool = True,
    pad_mode: str = "constant",
    cval: float | str = 0.0,
    **kwargs,
):
    """
    Load a Cartesian NIfTI, map it to spherical coords, then save:
      - the spherical tensor as NIfTI at `out_sph_path`
      - (optionally) the grid_meta dict as JSON at `out_meta_path`
    Returns the grid_meta.
    """
    # load
    vol, affine, header = load_nifti_file(in_path)

    if center is None:
        # take the cartesian center
        center = (vol.shape[0] / 2, vol.shape[1] / 2, vol.shape[2] / 2)
    elif isinstance(center, str):
        if center == "com":
            # compute the center of mass
            center = center_of_mass(vol)

    # forward transform
    sph, grid_meta = cartesian_to_spherical(
        vol,
        r_bins=r_bins,
        theta_bins=theta_bins,
        phi_bins=phi_bins,
        center=center,
        interp_order=interp_order,
        equal_area_phi=equal_area_phi,
        pad_mode=pad_mode,
        cval=cval,
    )

    # save spherical data
    # we use an identity affine since spherical coords have no standard voxel-to-world
    save_nifti_file(sph, np.eye(4), None, out_sph_path)

    # optionally dump metadata
    if out_meta_path:
        with open(out_meta_path, "w") as f:
            json.dump(
                {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in grid_meta.items()
                },
                f,
                indent=2,
            )

    return grid_meta


def spherical_to_nifti(
    in_path: str,
    out_cart_path: str,
    grid_meta_path: str,
    interp_order: int = 3,
    pad_mode: str = "constant",
    cval: float = 0.0,
):
    """
    Load a spherical NIfTI, map it to Cartesian coords, then save:
      - the Cartesian tensor as NIfTI at `out_cart_path`
      - (optionally) the grid_meta dict as JSON at `out_meta_path`
    Returns the grid_meta.
    """
    # load
    sph, affine, header = load_nifti_file(in_path)

    # load metadata
    with open(grid_meta_path, "r") as f:
        grid_meta = json.load(f)

    # inverse transform
    cart_shape = tuple(grid_meta["orig_shape"])  # <-- use the stored original shape
    vol = spherical_to_cartesian(
        sph,
        cart_shape=cart_shape,
        grid_meta=grid_meta,
        interp_order=interp_order,
        pad_mode=pad_mode,
        cval=cval,
    )

    # save Cartesian data
    save_nifti_file(vol, np.eye(4), None, out_cart_path)
