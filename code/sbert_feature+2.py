"""
Generate cumulative sentence embeddings (ALL LAYERS) from a word-level alignment CSV.
Context: Current sentence + Previous 2 sentences.

Input:
  /home/s-kawashima/research/sentence/{narrative}/sentence_alignment.csv

Output:
  /home/s-kawashima/research/output/features/{modelname}_cumulative_add_two-sentences/{narrative}.pkl
  /home/s-kawashima/research/output/features/{modelname}_cumulative_add_two-sentences/{narrative}.h5
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
    直前の2文の文脈を考慮して、単語ごとに prefix_text を作成
    [前々文] + [前文] + [現在の文の累積単語]
    """
    rows = []
    # 処理順序を保証
    df = df.sort_values(["sentence_id", "relative_start"]).reset_index(drop=True)

    # 過去の文を保持するリスト (最大2文)
    completed_sentences = [] 
    MAX_CONTEXT = 2

    for sid, g in df.groupby("sentence_id"):
        current_sentence_words = [] # 現在の文の累積単語
        
        # 過去の文脈を結合 (古い順に結合)
        global_context_prefix = " ".join(completed_sentences)
        if global_context_prefix:
            global_context_prefix += " " 

        # 現在の文の単語を順に処理
        for i, row in g.iterrows():
            token = str(row["normalized"])
            current_sentence_words.append(token)
            
            # 現在の文内での累積テキスト
            current_prefix = " ".join(current_sentence_words)
            
            # 最終的な prefix_text
            prefix_text = global_context_prefix + current_prefix
            
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

        # 現在の文の処理が終わったら、その全文を履歴に追加
        full_current_sent = " ".join(current_sentence_words)
        completed_sentences.append(full_current_sent)
        
        # 履歴がMAX_CONTEXTを超えたら最古のものを削除
        if len(completed_sentences) > MAX_CONTEXT:
            completed_sentences.pop(0)

    return pd.DataFrame(rows)


def mean_pooling(token_embeddings, attention_mask):
    """SBERTと同様のMean Pooling"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_all_layer_embeddings(model, texts, batch_size, device):
    """モデルの全層の埋め込みを取得する"""
    transformer = model[0].auto_model
    tokenizer = model.tokenizer
    transformer.config.output_hidden_states = True
    
    all_layer_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding layers"):
        batch_texts = texts[i : i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = transformer(**encoded_input)
        
        states = outputs.hidden_states
        batch_res = []
        for layer_tensor in states:
            pooled = mean_pooling(layer_tensor, encoded_input['attention_mask'])
            batch_res.append(pooled.cpu().numpy())
            
        batch_res = np.stack(batch_res, axis=1)
        all_layer_embeddings.append(batch_res)

    return np.concatenate(all_layer_embeddings, axis=0)


def main(narrative: str, modelname: str, batch_size: int, device: str):
    csv_path = f"/home/s-kawashima/research/output/sentence/{narrative}/sentence_alignments.csv"
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = ["sentence_id", "sentence_start", "word", "normalized", "relative_start", "relative_end", "absolute_start", "absolute_end"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    cumulative_df = build_cumulative_df(df)
    texts = cumulative_df["prefix_text"].tolist()
    print(f"#Cumulative samples: {len(texts)}")

    hfmodelname = HFMODELS.get(modelname, modelname)
    print(f"Loading model: {hfmodelname}")
    model = SentenceTransformer(hfmodelname, device=device)
    
    print("Encoding cumulative sentences (ALL LAYERS)...")
    embeddings_all_layers = get_all_layer_embeddings(model, texts, batch_size, device)
    
    print("Encoding sentence embeddings (SBERT final output)...")
    sentence_embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # 出力パス設定 (_add_two-sentences)
    epath = Path(
        root="/home/s-kawashima/research/output/features",
        datatype=f"{modelname}_cumulative_add_two-sentences",
        suffix=None,
        ext="pkl"
    )

    epath.update(narrative=narrative, ext="pkl")
    epath.mkdirs()
    cumulative_df.to_pickle(epath)
    print(f"Saved DataFrame → {epath}")

    epath.update(ext=".h5")
    with h5py.File(epath, "w") as f:
        f.create_dataset("hidden_states", data=embeddings_all_layers)
        f.create_dataset("sentence_embeddings", data=sentence_embeddings)
        f.attrs['n_layers'] = embeddings_all_layers.shape[1]
        f.attrs['dim'] = embeddings_all_layers.shape[2]
        f.attrs['model'] = hfmodelname

    print(f"Saved embeddings → {epath}")
    print("✅ Done (sbert+2).")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-n", "--narrative", required=True, help="Story name")
    parser.add_argument("-m", "--modelname", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(**vars(args))