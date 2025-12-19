"""
Combined Encoding: Whisper + SBERT (Cross-Narrative)
Merges both feature sets into a single MultipleKernelRidgeCV model
"""

from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Himalaya (Ridge Regression)
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, MultipleKernelRidgeCV, Kernelizer
from himalaya.scoring import correlation_score_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer

# Custom modules
from constants import SUBS, TR, TRS, NARRATIVE_SLICE
from get_bold import get_bold as load_bold_masked
from convert_clean_h5_to_pkl import get_bold_cached
from util.path import Path


# =============================================================================
# Feature Extraction: Whisper
# =============================================================================
def _get_whisper_embs(
    narrative: str, modelname: str, suffix: str = None, layer: int = None
):
    """Load Whisper embeddings and downsample to TR"""
    filename = f"/home/s-kawashima/research/output/features/{modelname}/narrative-{narrative}.pkl"
    df = pd.read_pickle(filename)
    df["start"] = df["start"].ffill()
    df["TR"] = df["start"].divide(TR[narrative]).apply(np.floor).apply(int)

    filename = f"/home/s-kawashima/research/output/features/{modelname}/narrative-{narrative}.h5"
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


def build_whisper_regressors(narrative: str, modelname: str, layer: int = None):
    """Build Whisper feature matrix (acoustic, encoder, decoder)"""
    conv_embs = _get_whisper_embs(narrative, modelname=modelname, suffix="_conv", layer=None)
    enc_embs = _get_whisper_embs(narrative, modelname=modelname, suffix="_enc", layer=None)
    dec_embs = _get_whisper_embs(narrative, modelname=modelname, suffix="_dec", layer=layer)

    X = np.hstack((conv_embs, enc_embs, dec_embs))

    slices = {}
    start = 0
    end = start + conv_embs.shape[1]
    slices["whisper_acoustic"] = slice(start, end)

    start = slices["whisper_acoustic"].stop
    end = start + enc_embs.shape[1]
    slices["whisper_encoder"] = slice(start, end)

    start = slices["whisper_encoder"].stop
    end = start + dec_embs.shape[1]
    slices["whisper_decoder"] = slice(start, end)

    return X, slices


# =============================================================================
# Feature Extraction: SBERT
# =============================================================================
def _get_sbert_features(narrative: str, modelname: str, layer: int):
    """
    Load SBERT cumulative embeddings and downsample to TR
    
    Args:
        layer: Layer index (0-based). Use -1 to get the final SBERT sentence embeddings
               (normalized, pooled output) instead of hidden states.
    """
    pkl_path = f"/home/s-kawashima/research/output/features/{modelname}_cumulative_add_one-sentence/narrative-{narrative}.pkl"
    print(f"Loading SBERT metadata: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    
    df["TR"] = df["absolute_start"].divide(TR[narrative]).apply(np.floor).apply(int)

    h5_path = f"/home/s-kawashima/research/output/features/{modelname}_cumulative_add_one-sentence/narrative-{narrative}.h5"
    
    with h5py.File(h5_path, "r") as f:
        if layer == -1:
            # Use final SBERT sentence embeddings (normalized, pooled)
            print(f"Loading SBERT sentence embeddings: {h5_path} (final output)")
            embeddings_raw = f["sentence_embeddings"][:]
        else:
            # Use hidden states from specific layer
            print(f"Loading SBERT hidden states: {h5_path} (Layer {layer})")
            data = f["hidden_states"]
            n_layers_saved = data.shape[1]
            
            if layer >= n_layers_saved:
                raise ValueError(f"Requested layer {layer} but file only has {n_layers_saved} layers.")
            embeddings_raw = data[:, layer, :]

    df["embedding"] = list(embeddings_raw)

    n_features = embeddings_raw.shape[1]
    n_trs = TRS[narrative]
    
    embeddings_tr = np.zeros((n_trs, n_features), dtype=np.float32)
    grouped = df.groupby("TR")["embedding"].apply(lambda x: np.mean(np.vstack(x), axis=0))
    
    for tr, emb in grouped.items():
        if 0 <= tr < n_trs:
            embeddings_tr[tr] = emb

    return embeddings_tr


def build_sbert_regressors(narrative: str, modelname: str, layer: int):
    """Build SBERT feature matrix"""
    sbert_embs = _get_sbert_features(narrative, modelname, layer)
    
    slices = {
        "sbert_cumulative": slice(0, sbert_embs.shape[1])
    }

    return sbert_embs, slices


# =============================================================================
# Combined Feature Building
# =============================================================================
def build_combined_regressors(
    narrative: str,
    whisper_model: str,
    whisper_layer: int,
    sbert_model: str,
    sbert_layer: int
):
    """Build combined feature matrix from Whisper and SBERT"""
    
    # Get Whisper features
    X_whisper, whisper_slices = build_whisper_regressors(narrative, whisper_model, whisper_layer)
    
    # Get SBERT features
    X_sbert, sbert_slices = build_sbert_regressors(narrative, sbert_model, sbert_layer)
    
    # Concatenate
    X = np.hstack((X_whisper, X_sbert))
    
    # Update SBERT slices to account for Whisper features
    offset = X_whisper.shape[1]
    combined_slices = {}
    
    # Add Whisper slices
    for name, sl in whisper_slices.items():
        combined_slices[name] = sl
    
    # Add SBERT slices with offset
    for name, sl in sbert_slices.items():
        combined_slices[name] = slice(sl.start + offset, sl.stop + offset)
    
    return X, combined_slices


def clean_features(X):
    """NaN handling and removal of constant features"""
    X = np.nan_to_num(X)
    std = X.std(axis=0)
    valid_cols = std > 1e-9
    if np.sum(~valid_cols) > 0:
        print(f"  Dropping {np.sum(~valid_cols)} constant features.")
        X = X[:, valid_cols]
    return X


def build_model(
    feature_names: list,
    slices: list,
    alphas: np.ndarray,
    n_jobs: int,
    verbose: int = 0,
    use_cuda: bool = False,
):
    """Build the MultipleKernelRidgeCV pipeline"""

    delayer_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Delayer(delays=[2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )

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


def encoding(
    whisper_model: str,
    whisper_layer: int,
    sbert_model: str,
    sbert_layer: int,
    alphas: np.ndarray,
    jobs: int,
    device: str,
    suffix: str,
    root: str,
    mask_path: str,
    pkl_dir: str = None,
    clean_dir: str = None,
    force_reload: bool = False,
    run: int = None,
    **kwargs,
):
    all_narratives = ["black", "forgot", "piemanpni", "bronx"]
    
    print(f"Whisper Model: {whisper_model} | Layer: {whisper_layer}")
    print(f"SBERT Model: {sbert_model} | Layer: {sbert_layer}")
    print(f"Device: {device}")

    # 1. Data preparation phase
    X_dict = {}
    Y_bold_dict = {}

    for narr in all_narratives:
        print(f"\nProcessing narrative: {narr}")
        
        # Build combined features
        X, feature_slices = build_combined_regressors(
            narr, whisper_model, whisper_layer, sbert_model, sbert_layer
        )
        X = clean_features(X)
        X_dict[narr] = X
        
        # Load BOLD data
        print(f"Loading BOLD data for {len(SUBS[narr])} subjects...")
        Y_bold_all = []
        
        for sub_id in tqdm(SUBS[narr], desc=f"Loading {narr} subjects"):
            if pkl_dir is not None:
                Y_bold_sub = get_bold_cached(
                    sub_id=sub_id,
                    narrative=narr,
                    output_dir=pkl_dir,
                    clean_root=clean_dir if clean_dir else "/home/s-kawashima/research/output/derivatives/clean",
                    force_reload=force_reload
                )
                if Y_bold_sub is None:
                    print(f"Warning: Failed to load cached data for {sub_id} {narr}. Skipping.")
                    continue
                Y_bold_sub = StandardScaler().fit_transform(Y_bold_sub)
            else:
                Y_bold_sub = load_bold_masked(
                    sub_id=sub_id, 
                    narrative=narr, 
                    root=root,
                    mask_path=mask_path,
                    run=run,
                    standardize=True, 
                    detrend=False
                )
                Y_bold_sub = Y_bold_sub[NARRATIVE_SLICE[narr]]

            Y_bold_all.append(Y_bold_sub)
        
        if not Y_bold_all:
            raise ValueError(f"No BOLD data loaded for narrative {narr}")
        
        Y_mean = np.mean(Y_bold_all, axis=0)
        Y_bold_dict[narr] = np.nan_to_num(Y_mean)
        print(f"Averaged BOLD shape for {narr}: {Y_bold_dict[narr].shape}")
        
        # Size check
        n_x = X_dict[narr].shape[0]
        n_y = Y_bold_dict[narr].shape[0]
        if n_x != n_y:
            print(f"  Warning: Shape mismatch X({n_x}) vs Y({n_y}). Trimming to min length.")
            min_len = min(n_x, n_y)
            X_dict[narr] = X_dict[narr][:min_len]
            Y_bold_dict[narr] = Y_bold_dict[narr][:min_len]

    # 2. Training & Prediction phase (Cross-Narrative)
    results = defaultdict(dict)
    use_cuda = (device == "cuda")
    
    # Feature names from the last processed narrative
    feature_names = list(feature_slices.keys())
    slices = list(feature_slices.values())

    print("\n" + "="*60)
    print("Starting cross-narrative predictions...")
    print("="*60)

    for train_narr in all_narratives:
        print(f"\n### Training on: {train_narr} ###")
        
        X_train = X_dict[train_narr]
        Y_train = Y_bold_dict[train_narr]
        
        pipeline = build_model(feature_names, slices, alphas, jobs, use_cuda=use_cuda)
        pipeline.fit(X_train, Y_train)
        
        mkr_model = pipeline[-1]
        best_alphas = mkr_model.best_alphas_
        print(f"Selected alphas min: {best_alphas.min():.2e}, max: {best_alphas.max():.2e}")
        
        for test_narr in all_narratives:
            print(f"  -> Predicting: {test_narr}")
            X_test = X_dict[test_narr]
            Y_test = Y_bold_dict[test_narr]
            
            Y_preds = pipeline.predict(X_test, split=True)
            scores = correlation_score_split(Y_test, Y_preds)
            
            key = f"{train_narr}->{test_narr}"
            
            def to_cpu(x):
                if hasattr(x, "detach"): return x.detach().cpu().numpy()
                return np.asarray(x)

            results[f"{key}_scores"] = to_cpu(scores)
            results[f"{key}_preds"] = to_cpu(Y_preds)
            results[f"{key}_actual"] = to_cpu(Y_test)

    # 3. Save results
    save_dir = f"/home/s-kawashima/research/output/results/encoding{suffix}"
    save_datatype = f"combined_whisper-{whisper_model}_layer-{whisper_layer}_sbert-{sbert_model}_layer-{sbert_layer}"
    
    out_path = Path(root=save_dir, sub="average", datatype=save_datatype, ext="h5")
    out_path.mkdirs()
    
    print(f"\nSaving to: {out_path}")
    with h5py.File(out_path, "w") as f:
        for k, v in results.items():
            f.create_dataset(k, data=v)
        # Save feature names for reference
        f.attrs["feature_names"] = feature_names
            
    print("Done.")


def main(**kwargs):
    if kwargs["device"] == "cuda":
        set_backend("torch_cuda")
        print("Set backend to torch cuda")
    
    encoding(**kwargs)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    # Whisper arguments
    parser.add_argument("--whisper-model", type=str, default="whisper-tiny")
    parser.add_argument("--whisper-layer", type=int, default=3)
    
    # SBERT arguments
    parser.add_argument("--sbert-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--sbert-layer", type=int, required=True)
    
    # Path arguments
    parser.add_argument("--root", type=str, required=True, help="BIDS derivatives root")
    parser.add_argument("--mask-path", type=str, required=True, help="Brain mask/atlas path")
    parser.add_argument("--pkl-dir", type=str, default=None, help="Cache directory for BOLD")
    parser.add_argument("--clean-dir", type=str, default="/home/s-kawashima/research/output/derivatives/clean")
    
    # Options
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--alphas", default=np.logspace(-5, 19, 25))
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--run", type=int, default=None)
    
    args = parser.parse_args()
    main(**vars(args))