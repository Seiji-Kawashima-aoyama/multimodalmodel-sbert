from os import makedirs, path
import h5py
import nibabel as nib
import numpy as np
from netneurotools import datasets as nntdata
from neuromaps.images import annot_to_gifti
import requests

DATADIR = "/home/s-kawashima/research/output/atlas"

# ディレクトリが存在しない場合は作成しておく
makedirs(DATADIR, exist_ok=True)

def get_brainmask():
    """Get a brain mask to remove the medial wall"""
    # 必要に応じてファイルをチェック・ダウンロードするロジックを入れても良いですが、
    # 今回は元のロジックを維持します
    lh_path = path.join(DATADIR, "lh.Medial_wall.label")
    rh_path = path.join(DATADIR, "rh.Medial_wall.label")
    
    # ファイルがない場合の簡易エラーハンドリング例（本来はここでDLすべき）
    if not path.exists(lh_path) or not path.exists(rh_path):
        print("Warning: Medial wall labels not found in 'mats'. Mask generation may fail.")

    lh_medial_indices = nib.freesurfer.io.read_label(lh_path)
    rh_medial_indices = nib.freesurfer.io.read_label(rh_path)
    fgmask = np.ones(81924, dtype=bool)
    fgmask[lh_medial_indices] = False
    fgmask[rh_medial_indices + 40962] = False
    return fgmask


class Atlas:
    def __init__(self, name: str, label_img: np.ndarray, labels: dict):
        self.name = name
        self.label_img = label_img
        self.id2label = labels
        self.label2id = {v: k for k, v in labels.items()}
        self.parcel_vcount = None
        # label2idのキーのリスト（背景0を除く）
        self.labels = list(self.label2id.keys())[1:]

    def label(self, key: int) -> str:
        return self.id2label.get(key)

    def key(self, label: str) -> int:
        return self.label2id.get(label)

    def num_voxels(self, label: str) -> int:
        if not self.parcel_vcount:
            uniques = np.unique(self.label_img, return_counts=True)
            self.parcel_vcount = {k: v for k, v in zip(*uniques)}
        return self.parcel_vcount[label]

    def __getitem__(self, key) -> str | int:
        if isinstance(key, int):
            return self.label(key)
        elif isinstance(key, str):
            return self.key(key)
        else:
            raise ValueError("key is incorrect type")

    def __len__(self) -> int:
        return len(self.id2label) - 1

    def vox_to_parc(
        self, values: np.ndarray, agg_func=np.mean, axis: int = -1
    ) -> np.ndarray:
        n_parcels = len(self)
        parcellation = self.label_img

        new_shape = list(values.shape)
        new_shape[axis] = n_parcels
        parcel_values = np.zeros(new_shape, dtype=values.dtype)
        for i in range(1, n_parcels + 1):
            parcel_mask = parcellation == i
            parcel_voxels = values[..., parcel_mask]
            parcel_values[..., i - 1] = agg_func(parcel_voxels, axis=-1)
        return parcel_values

    def parc_to_vox(self, values: np.ndarray) -> np.ndarray:
        parcellation = self.label_img
        new_shape = list(values.shape)
        new_shape[-1] = parcellation.size
        voxel_values = np.zeros(new_shape, dtype=values.dtype)
        for i in range(1, len(self) + 1):
            parcel_mask = parcellation == i
            voxel_values[..., parcel_mask] = values[..., i - 1 : i]

        return voxel_values

    def parcellate(self, values: np.ndarray, **kwargs) -> np.ndarray:
        return self.parc_to_vox(self.vox_to_parc(values, **kwargs))

    def get_background_mask(self) -> np.ndarray:
        return self.label_img == 0

    def roimask(self, rois: list[str | int]) -> np.ndarray:
        if len(rois) == 1:
            roi_id = rois[0]
            roi_id = self[roi_id] if isinstance(roi_id, str) else roi_id
            return self.label_img == roi_id
        else:
            roi_ids = np.array([self[r] if isinstance(r, str) else r for r in rois])
            return np.in1d(self.label_img, roi_ids)

    def to_network(self, symmetric: bool = False):
        network_img = np.zeros_like(self.label_img)
        start = 2 if symmetric else 1

        networks = np.unique(
            ["_".join(lb.split("_")[start:3]) for lb in self.labels]
        ).tolist()
        network2id = {v: k for k, v in enumerate(networks, 1)}
        network2id["_".join(self.label(0).split("_")[start:3])] = 0

        id2network = {}
        id2network[0] = self[0]
        id2network |= {k: v for k, v in enumerate(networks, 1)}

        for label, parc_id in self.label2id.items():
            parc_net = "_".join(label.split("_")[start:3])
            net_id = network2id[parc_net]
            network_img[self.label_img == parc_id] = net_id

        return Atlas(self.name, network_img, id2network)

    def save(self, data_dir=DATADIR) -> None:
        """AtlasをHDF5形式で保存する"""
        makedirs(data_dir, exist_ok=True)
        filepath = f"{data_dir}/{self.name}.hdf5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset(name="label_img", data=self.label_img)
            # 文字列リストを保存するためのエンコーディング処理
            f.create_dataset(name="ids", data=list(self.id2label.keys()))
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset(name="labels", data=list(self.id2label.values()), dtype=dt)
        print(f"Atlas saved to {filepath}")

    @staticmethod
    def load(atlas_name: str, data_dir=DATADIR):
        """HDF5からAtlasを読み込む"""
        filepath = f"{data_dir}/{atlas_name}.hdf5"
        if not path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist.")
            
        with h5py.File(filepath, "r") as f:
            label_img = f["label_img"][...]
            label_ids = f["ids"][...]
            label_names = f["labels"][...]
            
            # byte文字列からstrに戻す処理
            labels = {}
            for lb_id, lb_name in zip(label_ids, label_names):
                if isinstance(lb_name, bytes):
                    lb_name = lb_name.decode()
                labels[lb_id] = lb_name
                
        return Atlas(atlas_name, label_img, labels)

    @staticmethod
    def schaefer(
        parcels: int = 1000,
        networks: int = 17,
        kong: bool = False,
        space: str = "fsaverage6",
    ):
        """
        Schaeferアトラスを取得する。
        すでに保存済み(HDF5)ならそれをロードし、なければダウンロードして処理・保存する。
        """
        # 1. アトラス名の構築
        kong_str = "Kong2022_" if kong else ""
        atlasname = f"Schaefer2018_{parcels}Parcels_{kong_str}{networks}Networks"

        # 2. ローカルに保存済みかチェックし、あればロードして終了
        try:
            return Atlas.load(atlasname)
        except (FileNotFoundError, OSError):
            print(f"{atlasname} not found locally. Fetching from remote...")

        # 3. なければダウンロード処理を開始
        url = (
            "https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/"
            f"Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/{space}/label/"
        )

        filename = f"Schaefer2018_{parcels}Parcels_{kong_str}{networks}Networks_order.annot"
        filenames = ["lh." + filename, "rh." + filename]
        local_files = []
        
        makedirs(DATADIR, exist_ok=True)

        for fname in filenames:
            local_fn = path.join(DATADIR, fname)
            local_files.append(local_fn)
            if not path.isfile(local_fn):
                print(f"Downloading {fname}...")
                response = requests.get(url + fname)
                if not response.ok:
                    raise RuntimeError("Unable to retrieve file " + url + fname)
                with open(local_fn, "wb") as f:
                    f.write(response.content)

        # 4. GIFTI/AnnotをNumpy配列に変換
        gLh, gRh = annot_to_gifti(tuple(local_files))
        label_img = np.concatenate((gLh.agg_data(), gRh.agg_data()))
        labels = (
            gLh.labeltable.get_labels_as_dict() | gRh.labeltable.get_labels_as_dict()
        )

        # 5. インスタンス化して、HDF5として保存（次回利用のため）
        atlas = Atlas(atlasname, label_img, labels)
        atlas.save()

        return atlas

    # 他のメソッドは元のまま維持
    @staticmethod
    def glasser2016(symmetric: bool = False):
        atlasname = "glasser2016"
        # チェックロジックを追加しても良い
        
        gLh = nib.load(f"{DATADIR}/tpl-fsaverage6_hemi-L_desc-MMP_dseg.label.gii")
        gRh = nib.load(f"{DATADIR}/tpl-fsaverage6_hemi-R_desc-MMP_dseg.label.gii")
        left_data = gLh.agg_data()
        right_data = gRh.agg_data()

        labels = gLh.labeltable.get_labels_as_dict()

        if not symmetric:
            right_data[right_data > 0] += 180
            right_labels = gRh.labeltable.get_labels_as_dict()
            right_labels = {i + 180: v for i, v in right_labels.items() if i != 0}
            labels |= right_labels

        label_img = np.concatenate((left_data, right_data))
        labels = {k: v.removesuffix("_ROI") for k, v in labels.items()}

        return Atlas(atlasname, label_img, labels)

    @staticmethod
    def ev2010():
        from neuromaps.transforms import mni152_to_fsaverage

        gifL, gifR = mni152_to_fsaverage(
            f"{DATADIR}/allParcels_language_SN220.nii", method="nearest"
        )
        lang_atlas = np.concatenate((gifL.agg_data(), gifR.agg_data()))
        # Labels definition (shortened for brevity as in original)
        labels = ["???", "LH_IFGorb", "LH_IFG", "LH_MFG", "LH_AntTemp", "LH_PostTemp", "LH_AngG",
                  "RH_IFGorb", "RH_IFG", "RH_MFG", "RH_AntTemp", "RH_PostTemp", "RH_AngG"]
        id2label = {i: label for i, label in enumerate(labels)}
        return Atlas("langnet", lang_atlas, id2label)

    @staticmethod
    def lana2022():
        from neuromaps.transforms import fsaverage_to_fsaverage
        niiL = nib.load(f"{DATADIR}/LH_LanA_n804.nii.gz")
        niiR = nib.load(f"{DATADIR}/RH_LanA_n804.nii.gz")
        
        # Gifti conversion logic...
        gifL = nib.GiftiImage(darrays=(nib.gifti.gifti.GiftiDataArray(niiL.get_fdata().squeeze().astype(np.float32)),))
        gifR = nib.GiftiImage(darrays=(nib.gifti.gifti.GiftiDataArray(niiR.get_fdata().squeeze().astype(np.float32)),))
        
        gifLn, gifRn = fsaverage_to_fsaverage((gifL, gifR), "41k", hemi=("L", "R"), method="linear")
        resampled_data = np.concatenate((gifLn.agg_data(), gifRn.agg_data()))
        return Atlas("LanA", resampled_data, {})