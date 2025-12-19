"""Story encoding"""

from collections import defaultdict
import numpy.core.numeric
import sys
sys.modules['numpy._core.numeric'] = numpy.core.numeric
import h5py
import numpy as np
import pandas as pd
import torch
from constants import SUBS, TR, TRS, NARRATIVE_SLICE
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from himalaya.scoring import correlation_score_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from tqdm import tqdm
from get_bold import get_bold as load_bold_masked
from convert_clean_h5_to_pkl import get_bold_cached
# from util.atlas import Atlas
from util.path import Path
from voxelwise_tutorials.delayer import Delayer

def _get_llm_embs(
    narrative: str, modelname: str, suffix: str = None, layer: int = None
):

    filename = f"/home/s-kawashima/research/output/features/{modelname}-audio-sentence/narrative-{narrative}.pkl"
    df = pd.read_pickle(filename)
    
    # audio-sentence版ではabsolute_startカラムを使用
    if "absolute_start" in df.columns:
        df["start"] = df["absolute_start"]
    elif "start" not in df.columns:
        raise KeyError("Neither 'absolute_start' nor 'start' column found in dataframe")
    
    df["start"] = df["start"].ffill()
    df["TR"] = df["start"].divide(TR[narrative]).apply(np.floor).apply(int)

    filename = f"/home/s-kawashima/research/output/features/{modelname}-audio-sentence/narrative-{narrative}.h5"
    with h5py.File(filename, "r") as f:
        states = f[f"activations{suffix}"]
        if layer is not None:
            states = states[:, layer]
        else:
            states = states[...]
    df["embedding"] = [e for e in states]

    n_features = df.iloc[0].embedding.size
    embeddings = np.zeros((TRS[narrative], n_features), dtype=np.float32)
    for tr in range(TRS[narrative]):
        subdf = df[df.TR == tr]
        if len(subdf):
            embeddings[tr] = subdf.embedding.mean(0)

    return embeddings


def build_regressors(narrative: str, modelname: str, **kwargs):
    # word_onsets, word_rates = get_nuisance_regressors(narrative)
    conv_embs = _get_llm_embs(
        narrative, modelname=modelname, suffix="_conv", layer=None
    )
    enc_embs = _get_llm_embs(narrative, modelname=modelname, suffix="_enc", layer=None)
    dec_embs = _get_llm_embs(
        narrative, modelname=modelname, suffix="_dec", layer=kwargs.get("layer")
    )

    X = np.hstack(
        (
            # word_onsets.reshape(-1, 1),
            # word_rates.reshape(-1, 1),
            conv_embs,
            enc_embs,
            dec_embs,
        )
    )

    slices = {}
    # slices["nuisance"] = slice(0, 2)

    start = 0  # slices["nuisance"].stop
    end = start + conv_embs.shape[1]
    slices["acoustic"] = slice(start, end)

    start = slices["acoustic"].stop
    end = start + enc_embs.shape[1]
    slices["encoder"] = slice(start, end)

    start = slices["encoder"].stop
    end = start + dec_embs.shape[1]
    slices["decoder"] = slice(start, end)

    return X, slices

def clean_features(X):
    # 1. NaN/Inf を 0 に置換
    X = np.nan_to_num(X)
    
    # 2. 分散が0（値が変化しない）の特徴量を削除、あるいは小さなノイズを足す
    # ここでは単純に標準偏差が0の列があるか確認
    std = X.std(axis=0)
    valid_cols = std > 1e-10  # ほぼ0のものは除外
    
    if np.sum(~valid_cols) > 0:
        print(f"Warning: Dropping {np.sum(~valid_cols)} constant features.")
        X = X[:, valid_cols]
        
    return X


def build_model(
    feature_names: list[str],
    slices: list[slice],
    alphas: np.ndarray,
    n_jobs: int,
    verbose: int = 0,
    use_cuda: bool = False,
):
    """Build the pipeline"""

    # Set up modeling pipeline - StandardScaler removed to avoid CUDA/numpy conflicts
    # X data will be standardized before entering the pipeline
    delayer_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Delayer(delays=[2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )

    # Make kernelizer
    # When using CUDA, set n_jobs=1 to avoid multiprocessing issues with GPU tensors
    effective_n_jobs = 1 if use_cuda else n_jobs
    kernelizers_tuples = [
        (name, delayer_pipeline, slice_) for name, slice_ in zip(feature_names, slices)
    ]
    column_kernelizer = ColumnKernelizer(kernelizers_tuples, n_jobs=effective_n_jobs)

    params = dict(
        alphas=alphas,
        progress_bar=verbose,
        n_iter=100,
        diagonalize_method="svd",
    )
    mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver_params=params)
    pipeline = make_pipeline(
        column_kernelizer,
        mkr_model,
    )

    return pipeline


def get_average_sub(narrative: str):
    return get_sub_bold(narrative).mean(0)


def get_sub_bold(narrative: str):
    m = 81924
    n = len(SUBS[narrative])
    Y_bold = np.zeros((n, TRS[narrative], m), dtype=np.float32)
    for i, sub in enumerate(tqdm(SUBS[narrative])):
        Y_bold[i] = get_bold(sub, narrative)
    return Y_bold


def encoding(
    narrative: str,
    modelname: str,
    layer: int,
    alphas: list,
    jobs: int,
    group_sub: bool,
    folds: int,
    suffix: str,
    **kwargs,
):
    # Define all narratives
    all_narratives = ["black", "forgot", "piemanpni", "bronx"]
    
    # Validate input narrative
    if narrative not in all_narratives:
        raise ValueError(f"narrative must be one of {all_narratives}, got {narrative}")
    
    # Build regressors and BOLD data for all narratives
    print("Building regressors for all narratives...")
    X_dict = {}
    Y_bold_dict = {}
    
    for narr in all_narratives:
        print(f"\nProcessing narrative: {narr}")
        
        # Build regressors
        
        X, features = build_regressors(narr, modelname, layer=layer)
        X = clean_features(X)
        X_dict[narr] = X
        
        # Load and average BOLD data across all subjects
        print(f"Loading BOLD data for {len(SUBS[narr])} subjects...")
        Y_bold_all = []
        
        pkl_dir = kwargs.get("pkl_dir")
        for sub_id in tqdm(SUBS[narr], desc=f"Loading {narr} subjects"):
            
            # --- MODIFIED BLOCK STARTS: Use Cache vs Raw Loading ---
            if pkl_dir is not None:
                # Use cached pickle loading
                # Assumes get_bold_cached handles loading from pkl_dir
                # If the cache is task-specific (e.g. from clean.py H5), it is usually already sliced
                Y_bold_sub = get_bold_cached(
                    sub_id=sub_id,
                    narrative=narr,
                    output_dir=pkl_dir,
                    # Fallback to general root if 'clean_root' isn't explicitly passed
                    clean_root=kwargs.get("clean_dir", "/home/s-kawashima/research/output/derivatives/clean"), 
                    force_reload=kwargs.get("force_reload", False)
                )
                Y_bold_sub = StandardScaler().fit_transform(Y_bold_sub)

                if Y_bold_sub is None:
                    print(f"Warning: Failed to load cached data for {sub_id} {narr}. Skipping.")
                    continue
                    
            # --- MODIFIED BLOCK ENDS ---

            Y_bold_all.append(Y_bold_sub)
        
        if not Y_bold_all:
            raise ValueError(f"No BOLD data loaded for narrative {narr}")
        
        # Average across subjects
        Y_mean = np.mean(Y_bold_all, axis=0)
        # Y_bold_dict[narr] = zscore(Y_mean, axis=0)
        Y_bold_dict[narr] = np.nan_to_num(Y_mean)
        print(f"Averaged BOLD shape for {narr}: {Y_bold_dict[narr].shape}")
        
        # Ensure Y_bold matches X in number of samples (timepoints)
        n_samples = X_dict[narr].shape[0]
        if Y_bold_dict[narr].shape[0] != n_samples:
            print(f"Warning: Y_bold for {narr} has {Y_bold_dict[narr].shape[0]} TRs but X has {n_samples}. Trimming to match.")
            Y_bold_dict[narr] = Y_bold_dict[narr][:n_samples]
    
    # Prepare model
    feature_names = list(features.keys())
    slices = list(features.values())
    use_cuda = kwargs.get("device") == "cuda"
    
    # Convert to numpy for storage (handle both numpy and torch tensors)
    def to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)
    
    # Perform cross-narrative predictions
    results = defaultdict(dict)
    
    print("\n" + "="*80)
    print("Starting cross-narrative predictions...")
    print("="*80)
    
    for train_narr in all_narratives:
        print(f"\n### Training on: {train_narr} ###")
        
        # Build fresh pipeline for each training narrative
        pipeline = build_model(feature_names, slices, alphas, jobs, use_cuda=use_cuda)
        
        # Standardize training data
        X_train = X_dict[train_narr]
        Y_train = Y_bold_dict[train_narr]
        
        # Fit model on training narrative
        print(f"Fitting model on {train_narr}...")
        pipeline.fit(X_train, Y_train)
        mkr_model = pipeline[-1]
        best_alphas = mkr_model.best_alphas_
        print(f"Selected alphas min: {best_alphas.min()}, max: {best_alphas.max()}")
        
        # Predict on all other narratives
        for test_narr in all_narratives:
            # if train_narr == test_narr:
            #     # Skip self-prediction
            #     continue
            
            print(f"  Predicting on: {test_narr}")
            
            # Standardize test data
            X_test = X_dict[test_narr]
            Y_test = Y_bold_dict[test_narr]
            
            # Make predictions
            Y_preds = pipeline.predict(X_test, split=True)
            scores_split = correlation_score_split(Y_test, Y_preds)
            
            # Store results with key format: "train_narr->test_narr"
            key_prefix = f"{train_narr}->{test_narr}"
            results[f"{key_prefix}_actual"] = to_numpy(Y_test)
            results[f"{key_prefix}_scores"] = to_numpy(scores_split)
            results[f"{key_prefix}_preds"] = to_numpy(Y_preds)
    
    print("\n" + "="*80)
    print("All predictions completed!")
    print("="*80)
    
    # Save results
    sub_id = "average"  # identifier for averaged data
    pklpath = Path(
        root=f"/home/s-kawashima/research/output/results/encoding{suffix}",
        sub=sub_id,
        datatype=modelname + f"_layer-{layer}",
        ext="h5",
    )
    pklpath.mkdirs()
    
    print(f"\nSaving results to: {pklpath}")
    with h5py.File(pklpath, "w") as f:
        for key, value in results.items():
            f.create_dataset(name=key, data=value)
    
    print(f"\nResults saved successfully!")
    print(f"Total number of prediction pairs: {len([k for k in results.keys() if k.endswith('_scores')])}")


def main(*args, **kwargs):
    if kwargs["device"] == "cuda":
        set_backend("torch_cuda")
        print("Set backend to torch cuda")
    encoding(*args, **kwargs)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--modelname", type=str, default="whisper-tiny")
    parser.add_argument("-l", "--layer", type=int, default=3)
    parser.add_argument("-n", "--narrative", type=str, default="black",
                        help="Narrative to process (black/forgot/piemanpni/bronx). This parameter is still required but all narratives will be processed.")
    parser.add_argument("-s", "--suffix", type=str, default="")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("-k", "--folds", type=int, default=1)
    parser.add_argument("--alphas", default=np.logspace(-5, 10, 20))
    parser.add_argument("--group-sub", action="store_true")
    parser.add_argument("--root", type=str, required=True, default="/disk1/MRI-Data_in-use/20_narrativefMRI/10_ds002245-v.1.0.3_Hasson/derivatives/fmriprep_v25.1.4/",
                        help="BIDS derivatives root, e.g., .../derivatives/fmriprep_v25.1.4")
    parser.add_argument("--run", type=int, default=None,
                        help="Run number if present (e.g., 1).")
    parser.add_argument("--pkl-dir", type=str, default="/home/s-kawashima/research/output/parcellated",
                        help="Directory containing cached BOLD pickle files. If provided, will use cached data for faster loading.")
    parser.add_argument("--clean-dir", type=str, default="/home/s-kawashima/research/output/derivatives/clean",
                        help="Source directory for cleaned HDF5 files (used if cache needs to be regenerated).")
    parser.add_argument("--force-reload", action="store_true",
                        help="Force reload BOLD data from NIfTI files even if cache exists (only used with --pkl-dir).")

    main(**vars(parser.parse_args()))