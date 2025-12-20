"""
Sentence Transformer Cumulative Encoding (Cross-Narrative) for sbert+2 (2 previous sentences)
"""
from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import zscore
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, MultipleKernelRidgeCV, Kernelizer
from himalaya.scoring import correlation_score_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from constants import SUBS, TR, TRS, NARRATIVE_SLICE
from convert_clean_h5_to_pkl import get_bold_cached
from util.path import Path

def _get_sbert_features(narrative: str, modelname: str, layer: int = None, use_sentence_embedding: bool = False):
    # パス変更: _add_two-sentences
    pkl_path = f"/home/s-kawashima/research/output/features/{modelname}_cumulative_add_two-sentences/narrative-{narrative}.pkl"
    print(f"Loading metadata: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    
    df["TR"] = df["absolute_start"].divide(TR[narrative]).apply(np.floor).apply(int)

    # パス変更: _add_two-sentences
    h5_path = f"/home/s-kawashima/research/output/features/{modelname}_cumulative_add_two-sentences/narrative-{narrative}.h5"
    
    with h5py.File(h5_path, "r") as f:
        if use_sentence_embedding:
            print(f"Loading sentence_embeddings: {h5_path}")
            if "sentence_embeddings" not in f:
                raise KeyError(f"'sentence_embeddings' not found in {h5_path}")
            embeddings_raw = f["sentence_embeddings"][:]
        else:
            print(f"Loading hidden_states: {h5_path} (Layer {layer})")
            if "hidden_states" not in f:
                 raise KeyError(f"'hidden_states' not found in {h5_path}")
            data = f["hidden_states"]
            n_layers_saved = data.shape[1]
            if layer is None: raise ValueError("layer must be specified")
            if layer >= n_layers_saved: raise ValueError(f"Requested layer {layer} but file only has {n_layers_saved} layers.")
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

def build_regressors(narrative: str, modelname: str, layer: int = None, use_sentence_embedding: bool = False):
    sbert_embs = _get_sbert_features(narrative, modelname, layer, use_sentence_embedding)
    X = sbert_embs
    slices = {"sbert_cumulative": slice(0, sbert_embs.shape[1])}
    return X, slices

def clean_features(X):
    X = np.nan_to_num(X)
    std = X.std(axis=0)
    valid_cols = std > 1e-9
    if np.sum(~valid_cols) > 0:
        print(f"  Dropping {np.sum(~valid_cols)} constant features.")
        X = X[:, valid_cols]
    return X

def build_model(feature_names, slices, alphas, n_jobs, verbose=0, use_cuda=False):
    delayer_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Delayer(delays=[2, 3, 4, 5]),
        Kernelizer(kernel="linear"),
    )
    effective_n_jobs = 1 if use_cuda else n_jobs
    kernelizers_tuples = [(name, delayer_pipeline, slice_) for name, slice_ in zip(feature_names, slices)]
    column_kernelizer = ColumnKernelizer(kernelizers_tuples, n_jobs=effective_n_jobs)
    params = dict(alphas=alphas, progress_bar=verbose, n_iter=100, diagonalize_method="svd")
    mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver_params=params)
    return make_pipeline(column_kernelizer, mkr_model)

def encoding(modelname, layer, use_sentence_embedding, alphas, jobs, device, suffix, root, mask_path, pkl_dir=None, clean_dir=None, force_reload=False, run=None, **kwargs):
    all_narratives = ["black", "forgot", "piemanpni", "bronx"]
    layer_info = "sentence_embedding" if use_sentence_embedding else f"layer-{layer}"
    print(f"Model: {modelname} | Feature: {layer_info} | Device: {device}")

    X_dict = {}
    Y_bold_dict = {}

    for narr in all_narratives:
        print(f"\nProcessing narrative: {narr}")
        X, feature_slices = build_regressors(narr, modelname, layer, use_sentence_embedding)
        X = clean_features(X)
        X_dict[narr] = X
        
        print(f"Loading BOLD data for {len(SUBS[narr])} subjects...")
        Y_bold_all = []
        for sub_id in tqdm(SUBS[narr], desc=f"Loading {narr} subjects"):
            if pkl_dir is not None:
                Y_bold_sub = get_bold_cached(sub_id, narr, pkl_dir, clean_dir if clean_dir else "/home/s-kawashima/research/output/derivatives/clean", force_reload)
                if Y_bold_sub is None: continue
                Y_bold_sub = StandardScaler().fit_transform(Y_bold_sub)
            Y_bold_all.append(Y_bold_sub)
        
        if not Y_bold_all: raise ValueError(f"No BOLD data for {narr}")
        Y_mean = np.mean(Y_bold_all, axis=0)
        Y_bold_dict[narr] = np.nan_to_num(Y_mean)
        
        n_x, n_y = X_dict[narr].shape[0], Y_bold_dict[narr].shape[0]
        if n_x != n_y:
            min_len = min(n_x, n_y)
            X_dict[narr] = X_dict[narr][:min_len]
            Y_bold_dict[narr] = Y_bold_dict[narr][:min_len]

    results = defaultdict(dict)
    use_cuda = (device == "cuda")
    feature_names = ["sbert_cumulative"]
    slices = [slice(0, None)]

    print("\nStarting cross-narrative predictions...")
    for train_narr in all_narratives:
        print(f"\n### Training on: {train_narr} ###")
        pipeline = build_model(feature_names, slices, alphas, jobs, use_cuda=use_cuda)
        pipeline.fit(X_dict[train_narr], Y_bold_dict[train_narr])
        
        for test_narr in all_narratives:
            print(f"  -> Predicting: {test_narr}")
            Y_preds = pipeline.predict(X_dict[test_narr], split=True)
            scores = correlation_score_split(Y_bold_dict[test_narr], Y_preds)
            
            key = f"{train_narr}->{test_narr}"
            def to_cpu(x): return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            results[f"{key}_scores"] = to_cpu(scores)
            results[f"{key}_preds"] = to_cpu(Y_preds)
            results[f"{key}_actual"] = to_cpu(Y_bold_dict[test_narr])

    save_dir = f"/home/s-kawashima/research/output/results/encoding{suffix}"
    # 保存名変更: _add_two-sentences
    save_datatype = f"{modelname}_cumulative_{layer_info}_add_two-sentences"
    
    out_path = Path(root=save_dir, sub="average", datatype=save_datatype, ext="h5")
    out_path.mkdirs()
    print(f"\nSaving to: {out_path}")
    with h5py.File(out_path, "w") as f:
        for k, v in results.items(): f.create_dataset(k, data=v)
        f.attrs['modelname'] = modelname
        f.attrs['use_sentence_embedding'] = use_sentence_embedding
        if not use_sentence_embedding: f.attrs['layer'] = layer
    print("Done.")

def main(**kwargs):
    if kwargs["device"] == "cuda": set_backend("torch_cuda")
    encoding(**kwargs)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-m", "--modelname", default="all-MiniLM-L6-v2")
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("-l", "--layer", type=int, help="Layer index (0-based)")
    feature_group.add_argument("--use-sentence-embedding", action="store_true", help="Use SBERT final sentence embedding")
    parser.add_argument("--root", type=str, required=True, help="BIDS derivatives root")
    parser.add_argument("--mask-path", type=str, required=True, help="Brain mask/atlas path")
    parser.add_argument("--pkl-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--clean-dir", type=str, default="/home/s-kawashima/research/output/derivatives/clean")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--alphas", default=np.logspace(0, 19, 20))
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()
    main(**vars(args))