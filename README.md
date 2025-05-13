# Cartesian ↔ Spherical NIfTI Transformations

This project provides utilities for converting 3D medical images (in NIfTI format) between Cartesian and spherical coordinate systems. It is useful for neuroimaging, medical image analysis, and geometric deep learning tasks where spherical representations are beneficial.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features
- Convert NIfTI volumes from Cartesian to spherical coordinates and back
- Support for equal-area latitude binning
- Flexible center selection (center of mass, geometric center, or custom)
- Metadata export for reproducible inverse transforms

## Installation

Clone this repository and install the requirements:

```bash
git clone https://github.com/mahbodez/cartesian2spherical.git
cd cartesian2spherical
pip install -r requirements.txt
```

## Quick Start

A demo script (`demo.py`) is provided:

```python
from spherical.transforms import (
    nifti_to_spherical,
    spherical_to_nifti,
)

# Convert a NIfTI file to spherical coordinates
nifti_to_spherical(
    './mri_14725586.nii.gz',
    './output_spherical.nii',
    out_meta_path='./output_meta.json',
    r_bins=224,
    theta_bins=224,
    phi_bins=224,
    center='com',  # use center of mass
    interp_order=5,
    equal_area_phi=True,
    pad_mode='constant',
    cval='min',
)

# Convert back to Cartesian
spherical_to_nifti(
    './output_spherical.nii',
    './output_cartesian.nii',
    grid_meta_path='./output_meta.json',
    interp_order=5,
    pad_mode='constant',
    cval='min',
)
```

## API Overview
- `nifti_to_spherical(in_path, out_sph_path, out_meta_path, ...)`  
  Converts a NIfTI file to spherical coordinates and saves the result and metadata.
- `spherical_to_nifti(in_path, out_cart_path, grid_meta_path, ...)`  
  Converts a spherical NIfTI file back to Cartesian coordinates using metadata.

See [`spherical/transforms.py`](spherical/transforms.py) for more details and options.

## Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Reporting Issues
If you encounter a bug or have a feature request, please use the [issue templates](.github/ISSUE_TEMPLATE/) provided.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
