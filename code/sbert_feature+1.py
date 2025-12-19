"""
Generate cumulative sentence embeddings (ALL LAYERS) from a word-level alignment CSV.

Input:
  /home/s-kawashima/research/sentence/{narrative}/sentence_alignment.csv

Output:
  /home/s-kawashima/research/output/features/{modelname}_cumulative/{narrative}.pkl
  /home/s-kawashima/research/output/features/{modelname}_cumulative/{narrative}.h5
"""

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from util.path import Path

HFMODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}


def build_cumulative_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    直前の文脈 (一つ前の文の全文) を考慮して、単語ごとに prefix_text を作成

    [一つ前の文の全文] + [現在の文の累積単語]
    """
    rows = []
    # 処理順序を保証
    df = df.sort_values(["sentence_id", "relative_start"]).reset_index(drop=True)

    # 直前の文の全文を保持する変数 (最初は空)
    last_completed_sentence = "" 

    for sid, g in df.groupby("sentence_id"):
        current_sentence_words = [] # 現在の文の累積単語
        
        # 過去の文脈として、直前の文の全文を使用
        global_context_prefix = last_completed_sentence 
        if global_context_prefix:
            # 過去文脈があれば、現在の累積単語との間にスペースを追加
            global_context_prefix += " " 

        # 現在の文の単語を順に処理
        for i, row in g.iterrows():
            token = str(row["normalized"])
            current_sentence_words.append(token)
            
            # 現在の文内での累積テキスト
            current_prefix = " ".join(current_sentence_words)
            
            # 最終的な prefix_text = [直前の文の全文] + [現在の文の累積単語]
            prefix_text = global_context_prefix + current_prefix
            
            # DataFrameに追加する情報を構築
            rows.append({
                "sentence_id": sid,
                "token_index": i,
                "word": row["word"],
                "normalized": row["normalized"],
                "prefix_text": prefix_text,
                "sentence_start": row["sentence_start"],
                "relative_start": row["relative_start"],
                "relative_end": row["relative_end"],
                "absolute_start": row["absolute_start"],
                "absolute_end": row["absolute_end"],
            })

        # 現在の文の処理が終わったら、その全文を last_completed_sentence に保存
        last_completed_sentence = " ".join(current_sentence_words)

    return pd.DataFrame(rows)


def mean_pooling(token_embeddings, attention_mask):
    """
    SBERTと同様のMean Poolingを手動で行う
    token_embeddings: (Batch, SeqLen, Dim)
    attention_mask: (Batch, SeqLen)
    return: (Batch, Dim)
    """
    # 修正箇所: unsqueeze(-1] -> unsqueeze(-1)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_all_layer_embeddings(model, texts, batch_size, device):
    """
    モデルの全層の埋め込みを取得する
    Return shape: (Num_Samples, Num_Layers, Dim)
    """
    # SBERTモデルの内部コンポーネントを取得
    transformer = model[0].auto_model
    tokenizer = model.tokenizer
    
    # 隠れ層を出力するように設定
    transformer.config.output_hidden_states = True
    
    all_layer_embeddings = []
    
    # バッチ処理
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding layers"):
        batch_texts = texts[i : i + batch_size]
        
        # Tokenize
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = transformer(**encoded_input)
        
        # outputs.hidden_states -> (Batch, Layers, Seq, Dim) -> Pooling -> (Batch, Layers, Dim)
        states = outputs.hidden_states
        
        batch_res = []
        for layer_tensor in states:
            # layer_tensor shape: (Batch, SeqLen, Dim)
            pooled = mean_pooling(layer_tensor, encoded_input['attention_mask'])
            batch_res.append(pooled.cpu().numpy())
            
        # batch_res structure: List of (Batch, Dim) -> Stack to (Batch, Layers, Dim)
        batch_res = np.stack(batch_res, axis=1)
        all_layer_embeddings.append(batch_res)

    # Concat all batches: (Total_Samples, Layers, Dim)
    return np.concatenate(all_layer_embeddings, axis=0)


def main(narrative: str, modelname: str, batch_size: int, device: str):
    # パスを自動生成
    csv_path = f"/home/s-kawashima/research/output/sentence/{narrative}/sentence_alignments.csv"
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # 必要列の確認
    required_cols = [
        "sentence_id", "sentence_start", "word", "normalized",
        "relative_start", "relative_end", "absolute_start", "absolute_end"
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # 累積テキスト作成
    cumulative_df = build_cumulative_df(df)
    texts = cumulative_df["prefix_text"].tolist()
    print(f"#Cumulative samples: {len(texts)}")

    # モデル読み込み
    hfmodelname = HFMODELS.get(modelname, modelname)
    print(f"Loading model: {hfmodelname}")
    model = SentenceTransformer(hfmodelname, device=device)
    emb_dim = model.get_sentence_embedding_dimension()
    print(f"Base Embedding dim: {emb_dim}")

    # エンコード (全層の隠れ状態)
    print("Encoding cumulative sentences (ALL LAYERS)...")
    embeddings_all_layers = get_all_layer_embeddings(model, texts, batch_size, device)
    print(f"Hidden states shape: {embeddings_all_layers.shape}")  # (N, L, D)

    # エンコード (正規の文埋め込み - SBERTの最終出力)
    print("Encoding sentence embeddings (SBERT final output)...")
    sentence_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2正規化を適用
    )
    print(f"Sentence embeddings shape: {sentence_embeddings.shape}")  # (N, D)

    # 出力パス設定
    epath = Path(
        root="/home/s-kawashima/research/output/features",
        datatype=f"{modelname}_cumulative_add_one-sentence",
        suffix=None,
        ext="pkl"
    )

    epath.update(narrative=narrative, ext="pkl")
    epath.mkdirs()
    cumulative_df.to_pickle(epath)
    print(f"Saved DataFrame → {epath}")

    epath.update(ext=".h5")
    with h5py.File(epath, "w") as f:
        # 全層の隠れ状態
        f.create_dataset("hidden_states", data=embeddings_all_layers)
        # 正規の文埋め込み（SBERT最終出力）
        f.create_dataset("sentence_embeddings", data=sentence_embeddings)
        # メタデータ
        f.attrs['n_layers'] = embeddings_all_layers.shape[1]
        f.attrs['dim'] = embeddings_all_layers.shape[2]
        f.attrs['model'] = hfmodelname

    print(f"Saved embeddings → {epath}")
    print("  - hidden_states: 全層の隠れ状態 (N, Layers, Dim)")
    print("  - sentence_embeddings: SBERT文埋め込み (N, Dim)")
    print("✅ Done.")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-n", "--narrative", required=True, help="Story name (e.g., pieman)")
    parser.add_argument("-m", "--modelname", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(**vars(args))