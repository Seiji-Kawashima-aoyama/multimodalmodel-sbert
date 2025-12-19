import nibabel as nib
import numpy as np
from util.path import Path


def get_bold_path(root: str, space: str, **kwargs) -> Path:
    bold_path = Path(
        root=root,
        datatype="func",
        sub="",
        task="",
        run="",
        hemi="L",
        space=space,
        desc="",
        suffix="bold",
        ext="",
    )

    if space.startswith("fs"):
        bold_path.update(hemi=kwargs.get("hemi", "L"), ext=".func.gii")
    elif space.startswith("MNI"):
        bold_path.update(ext=".nii.gz")
        del bold_path["hemi"]

    if kwargs.get("numpy", False):
        bold_path.update(ext=".npy")

    if desc := kwargs.get("desc"):
        bold_path.update(desc=desc)
    else:
        del bold_path["desc"]

    return bold_path


def load_bold(file_path: str | Path) -> np.ndarray:
    if isinstance(file_path, Path):
        file_path = file_path.fpath

    if file_path.endswith("npy"):
        return np.load(file_path)
    elif file_path.endswith("gii"):
        return nib.load(file_path).agg_data()
    elif file_path.endswith("nii.gz"):
        return nib.load(file_path).get_fdata()
    else:
        raise ValueError(f"Unknown file type: {file_path}")
