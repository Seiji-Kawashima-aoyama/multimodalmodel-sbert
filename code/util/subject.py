import nibabel as nib
import numpy as np
from util.path import Path
import nibabel as nib
import numpy as np
from nilearn import signal
from sklearn.base import BaseEstimator, TransformerMixin

from .path import Path


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


class GiftiMasker(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.init_args = kwargs

    def fit(self, gifti_imgs: Path | list[Path], **kwargs):
        self.gifti_img = gifti_imgs
        self.init_args.update(kwargs)
        return self

    def transform(self, gifti_imgs: Path | list[Path]):
        if not isinstance(gifti_imgs, list):
            gifti_imgs = [gifti_imgs]

        images = []
        for gifti_img in gifti_imgs:
            gifti = nib.load(gifti_img)
            signals = gifti.agg_data().T  # type:ignore
            images.append(signal.clean(signals, **self.init_args))

        signals = np.hstack(images)

        return signals