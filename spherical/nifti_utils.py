import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output


def load_nifti_file(filepath: str):
    """
    Load a NIfTI file and return (data array, affine, header).
    """
    img = nib.load(filepath)
    img = resample_to_output(img, voxel_sizes=(1.0, 1.0, 1.0), order=3)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def save_nifti_file(
    data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header | None, out_path: str
):
    """
    Save `data` as a NIfTI to `out_path`. If header is provided it is copied.
    """
    hdr = header.copy() if header is not None else None
    out_img = nib.Nifti1Image(data, affine, header=hdr)
    nib.save(out_img, out_path)
