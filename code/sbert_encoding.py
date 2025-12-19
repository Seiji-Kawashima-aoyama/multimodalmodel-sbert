"""
Sentence Transformer Cumulative Encoding (Cross-Narrative)

Modified to support:
  - hidden_states: 全層の隠れ状態 (N, Layers, Dim)
  - sentence_embeddings: SBERT文埋め込み (N, Dim)
"""

from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import zscore

# Himalaya (Ridge Regression)
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, MultipleKernelRidgeCV, Kernelizer
from himalaya.scoring import correlation_score_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer

# Custom modules (既存の環境に合わせてパスが通っている前提)
from constants import SUBS, TR, TRS, NARRATIVE_SLICE
from convert_clean_h5_to_pkl import get_bold_cached
from util.path import Path


def _get_sbert_features(narrative: str, modelname: str, layer: int = None, use_sentence_embedding: bool = False):
    """
    Sentence Transformerの累積埋め込みを読み込み、TRごとに平均化する
    
    Parameters
    ----------
    narrative : str
        物語名
    modelname : str
        モデル名
    layer : int, optional
        使用するレイヤーのインデックス (0-based)。use_sentence_embedding=Falseの場合に必要
    use_sentence_embedding : bool
        Trueの場合、SBERTの正規文埋め込みを使用。Falseの場合、hidden_statesの指定レイヤーを使用
    
    Returns
    -------
    embeddings_tr : np.ndarray
        TRごとに平均化された埋め込み (n_trs, dim)
    """
    # 1. メタデータ(DataFrame)の読み込み
    pkl_path = f"/home/s-kawashima/research/output/features/{modelname}_cumulative/narrative-{narrative}.pkl"
    print(f"Loading metadata: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    
    # 時間情報の処理 (absolute_start を使用)
    # TRインデックスを計算 (秒数 / TR秒数)
    df["TR"] = df["absolute_start"].divide(TR[narrative]).apply(np.floor).apply(int)

    # 2. 埋め込みベクトル(HDF5)の読み込み
    h5_path = f"/home/s-kawashima/research/output/features/{modelname}_cumulative/narrative-{narrative}.h5"
    
    with h5py.File(h5_path, "r") as f:
        if use_sentence_embedding:
            # 正規のSBERT文埋め込みを使用
            print(f"Loading sentence_embeddings: {h5_path}")
            if "sentence_embeddings" not in f:
                raise KeyError(f"'sentence_embeddings' not found in {h5_path}. "
                              "Please regenerate the file with the updated sbert_feature.py")
            embeddings_raw = f["sentence_embeddings"][:]  # shape: (N_samples, Dim)
        else:
            # hidden_statesの指定レイヤーを使用
            print(f"Loading hidden_states: {h5_path} (Layer {layer})")
            if "hidden_states" not in f:
                # 後方互換性: 古い形式のファイルをサポート
                if "cumulative_embeddings" in f:
                    print("  (Using legacy 'cumulative_embeddings' dataset)")
                    data = f["cumulative_embeddings"]
                else:
                    raise KeyError(f"Neither 'hidden_states' nor 'cumulative_embeddings' found in {h5_path}")
            else:
                data = f["hidden_states"]
            
            n_layers_saved = data.shape[1]
            
            if layer is None:
                raise ValueError("layer must be specified when use_sentence_embedding=False")
            if layer >= n_layers_saved:
                raise ValueError(f"Requested layer {layer} but file only has {n_layers_saved} layers.")
                
            # 指定されたレイヤーのみ取得: (N_samples, Dim)
            embeddings_raw = data[:, layer, :]

    # DataFrameに一時的に格納してgroupbyしやすくする
    df["embedding"] = list(embeddings_raw)

    # 3. TRごとの平均化 (Downsampling)
    n_features = embeddings_raw.shape[1]
    n_trs = TRS[narrative]
    
    embeddings_tr = np.zeros((n_trs, n_features), dtype=np.float32)

    # TRごとにグループ化して平均をとる
    grouped = df.groupby("TR")["embedding"].apply(lambda x: np.mean(np.vstack(x), axis=0))
    
    for tr, emb in grouped.items():
        if 0 <= tr < n_trs:
            embeddings_tr[tr] = emb

    return embeddings_tr


def build_regressors(narrative: str, modelname: str, layer: int = None, use_sentence_embedding: bool = False):
    """特徴量行列 X を作成する"""
    
    # SBERTの特徴量を取得
    sbert_embs = _get_sbert_features(narrative, modelname, layer, use_sentence_embedding)
    
    # 必要ならここで他の特徴量(単語レートなど)と結合するが、
    # 今回は純粋にSentence Transformerの効果を見るためこれ単体とする
    X = sbert_embs

    # Himalaya用のスライス情報
    slices = {
        "sbert_cumulative": slice(0, sbert_embs.shape[1])
    }

    return X, slices


def clean_features(X):
    """NaN処理と無変化特徴量の削除"""
    X = np.nan_to_num(X)
    std = X.std(axis=0)
    # 分散がほぼ0の特徴量は削除
    valid_cols = std > 1e-9
    if np.sum(~valid_cols) > 0:
        print(f"  Dropping {np.sum(~valid_cols)} constant features.")
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


def encoding(
    modelname: str,
    layer: int,
    use_sentence_embedding: bool,
    alphas: list,
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
    # 対象とする物語
    all_narratives = ["black", "forgot", "piemanpni", "bronx"]
    
    # 表示用のレイヤー情報
    if use_sentence_embedding:
        layer_info = "sentence_embedding"
    else:
        layer_info = f"layer-{layer}"
    
    print(f"Model: {modelname} | Feature: {layer_info} | Device: {device}")

    # 1. データ準備フェーズ
    X_dict = {}
    Y_bold_dict = {}

    for narr in all_narratives:
        print(f"\nProcessing narrative: {narr}")
        
        # 特徴量 X の構築
        X, feature_slices = build_regressors(narr, modelname, layer, use_sentence_embedding)
        X = clean_features(X)
        X_dict[narr] = X
        
        # 脳活動データ Y の読み込み (被験者平均)
        print(f"Loading BOLD data for {len(SUBS[narr])} subjects...")
        Y_bold_all = []
        
        for sub_id in tqdm(SUBS[narr], desc=f"Loading {narr} subjects"):
            # --- MODIFIED BLOCK: Use Cache vs Raw Loading (whisper版に合わせて修正) ---
            if pkl_dir is not None:
                # Use cached pickle loading
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

            Y_bold_all.append(Y_bold_sub)
        
        if not Y_bold_all:
            raise ValueError(f"No BOLD data loaded for narrative {narr}")
        
        # 被験者平均
        Y_mean = np.mean(Y_bold_all, axis=0)
        Y_bold_dict[narr] = np.nan_to_num(Y_mean)
        print(f"Averaged BOLD shape for {narr}: {Y_bold_dict[narr].shape}")
        
        # サイズ整合性チェック
        n_x = X_dict[narr].shape[0]
        n_y = Y_bold_dict[narr].shape[0]
        if n_x != n_y:
            print(f"  Warning: Shape mismatch X({n_x}) vs Y({n_y}). Trimming to min length.")
            min_len = min(n_x, n_y)
            X_dict[narr] = X_dict[narr][:min_len]
            Y_bold_dict[narr] = Y_bold_dict[narr][:min_len]

    # 2. 学習・予測フェーズ (Cross-Narrative)
    results = defaultdict(dict)
    use_cuda = (device == "cuda")
    
    # 特徴量名
    feature_names = ["sbert_cumulative"]
    slices = [slice(0, None)]

    print("\n" + "="*60)
    print("Starting cross-narrative predictions...")
    print("="*60)

    for train_narr in all_narratives:
        print(f"\n### Training on: {train_narr} ###")
        
        X_train = X_dict[train_narr]
        Y_train = Y_bold_dict[train_narr]
        
        # パイプライン構築 & 学習
        pipeline = build_model(feature_names, slices, alphas, jobs, use_cuda)
        pipeline.fit(X_train, Y_train)
        
        # 各テスト物語に対して予測
        for test_narr in all_narratives:
            # if train_narr == test_narr: continue

            print(f"  -> Predicting: {test_narr}")
            X_test = X_dict[test_narr]
            Y_test = Y_bold_dict[test_narr]
            
            # 予測
            Y_preds = pipeline.predict(X_test, split=True)
            scores = correlation_score_split(Y_test, Y_preds)
            
            # 結果格納 (numpy化)
            key = f"{train_narr}->{test_narr}"
            
            # Tensor対策
            def to_cpu(x):
                if hasattr(x, "detach"): return x.detach().cpu().numpy()
                return np.asarray(x)

            results[f"{key}_scores"] = to_cpu(scores)
            results[f"{key}_preds"] = to_cpu(Y_preds)
            results[f"{key}_actual"] = to_cpu(Y_test)

    # 3. 結果保存
    save_dir = f"/home/s-kawashima/research/output/results/encoding{suffix}"
    save_datatype = f"{modelname}_cumulative_{layer_info}"
    
    out_path = Path(root=save_dir, sub="average", datatype=save_datatype, ext="h5")
    out_path.mkdirs()
    
    print(f"\nSaving to: {out_path}")
    with h5py.File(out_path, "w") as f:
        for k, v in results.items():
            f.create_dataset(k, data=v)
        # メタデータ保存
        f.attrs['modelname'] = modelname
        f.attrs['use_sentence_embedding'] = use_sentence_embedding
        if not use_sentence_embedding:
            f.attrs['layer'] = layer
            
    print("Done.")


def main(**kwargs):
    if kwargs["device"] == "cuda":
        set_backend("torch_cuda")
    
    encoding(**kwargs)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    # 必須・主要な引数
    parser.add_argument("-m", "--modelname", default="all-MiniLM-L6-v2")
    
    # 特徴量選択 (排他的オプション)
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("-l", "--layer", type=int,
                               help="Layer index (0-based) to use from hidden_states")
    feature_group.add_argument("--use-sentence-embedding", action="store_true",
                               help="Use SBERT's final sentence embedding instead of hidden_states")
    
    # パス周り (環境に合わせて指定してください)
    parser.add_argument("--root", type=str, required=True, help="BIDS derivatives root")
    parser.add_argument("--mask-path", type=str, required=True, help="Brain mask/atlas path")
    parser.add_argument("--pkl-dir", type=str, default=None, help="Cache directory for BOLD")
    parser.add_argument("--clean-dir", type=str, default="/home/s-kawashima/research/output/derivatives/clean",
                        help="Source directory for cleaned HDF5 files (used if cache needs to be regenerated).")
    
    # オプション
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--alphas", default=np.logspace(0, 19, 20))
    parser.add_argument("--force-reload", action="store_true")
    
    args = parser.parse_args()
    main(**vars(args))
