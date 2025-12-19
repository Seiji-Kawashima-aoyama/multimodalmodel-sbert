"""Run confound regression on fMRI-prepped BOLD signal."""

import warnings

import h5py
import numpy as np
import pandas as pd
from constants import CONFOUND_MODEL, NARRATIVE_SLICE, SUBS, TR, TRS
from extract_confounds import extract_confounds, load_confounds
from tqdm import tqdm
from util import subject
from util.path import Path


def get_nuisance_regressors(narrative: str):
    # 1. パスの指定
    # {narrative} の部分に "pieman" などが入ります。
    # もしファイル名に拡張子(.csv)が必要なら末尾に追加してください。
    filename = f"/disk1/MRI-Data_in-use/20_narrativefMRI/10_ds002245-v.1.0.3_Hasson/stimuli/gentle/{narrative}/align.csv"

    # 2. CSVの読み込み（重要修正）
    # ヘッダーがないファイルなので header=None とし、列名を手動で付けます
    # index_col=0 は削除します
    df = pd.read_csv(filename, header=None, names=['word', 'lemma', 'onset', 'offset'])

    # 3. 欠損値の処理（念のため）
    # onset（開始時間）が入っていない行（句読点など）を除外します
    df = df.dropna(subset=['onset'])

    # 4. 秒数(onset) を TR番号 に変換（重要修正）
    # インポートした定数 TR (例: 1.5) で割って整数にします
    df['TR'] = (df['onset'] / TR[narrative]).astype(int)

    # 物語に対応する総TR数を取得
    n_trs = TRS[narrative]

    # 結果格納用の配列を初期化
    word_onsets = np.zeros(n_trs, dtype=np.float32)
    word_rates = np.zeros(n_trs, dtype=np.float32)

    # 5. TRごとに集計
    for tr in range(n_trs):
        # ここで、上で計算した 'TR' 列を使ってデータを抽出します
        subdf = df[df.TR == tr]

        # 該当するTRに単語が存在すればフラグを立て、単語数を記録
        if len(subdf) > 0:
            word_onsets[tr] = 1
            word_rates[tr] = len(subdf)

    return word_onsets, word_rates


def get_bold(sub: int, narrative: str) -> np.ndarray:
    boldpath = Path(
        root="/mnt/s23_disk3/MRI-Data/01_data/20_narrativefMRI/10_ds002245-v.1.0.3_Hasson/derivatives/fmriprep_v25.1.4/",
        datatype="func",
        sub=f"{sub:03d}",
        task=narrative,
        hemi="L",
        space="fsaverage6",
        suffix="bold",
        ext=".func.gii",
    )
    boldpath.update(sub=f"{sub:03d}")
    paths = [boldpath, boldpath.copy().update(hemi="R")]

    confpath = Path(
        root="/disk1/MRI-Data_in-use/20_narrativefMRI/10_ds002245-v.1.0.3_Hasson/derivatives/fmriprep/",
        datatype="func",
        sub=f"{sub:03d}",
        task=narrative,
        space="fsaverage6",
        hemi="L",
        ext=".func.gii",
    )
    del confpath["hemi"]
    del confpath["space"]
    confpath.update(desc="confounds", suffix="regressors", ext=".tsv")

    confounds_fn = confpath.fpath
    confounds_df, confounds_meta = load_confounds(confounds_fn)
    confounds = extract_confounds(confounds_df, confounds_meta, CONFOUND_MODEL)

    word_onsets, word_rates = get_nuisance_regressors(narrative)
    task_confounds = np.zeros((len(confounds), 2))
    task_confounds[NARRATIVE_SLICE[narrative], 0] = word_onsets
    task_confounds[NARRATIVE_SLICE[narrative], 1] = word_rates

    all_confounds = np.hstack((confounds.to_numpy(), task_confounds))

    masker = subject.GiftiMasker(
        t_r=TR[narrative],
        detrend=True,
        ensure_finite=True,
        standardize="zscore_sample",
        standardize_confounds=True,
    )
    Y_bold = masker.fit_transform(paths, confounds=all_confounds)

    Y_bold = Y_bold[NARRATIVE_SLICE[narrative]]

    return Y_bold


def main(narratives: str, subject_id: int, **kwargs):
    # subject_id は引数から受け取った単一のID (int) です
    for narrative in narratives:

        # tqdmループも削除し、受け取った subject_id をそのまま使用
        sub_id = subject_id

        print(f"Processing Subject: {sub_id}, Narrative: {narrative}") # ログ用に追加

        boldpath = Path(
            root="/home/s-kawashima/research/output/derivatives/clean",
            datatype="func",
            sub=f"{sub_id:03d}",
            task=narrative,
            space="fsaverage6",
            ext=".h5",
        )
        boldpath.mkdirs()

        try:
            Y_bold = get_bold(sub_id, narrative)
            print(Y_bold.shape)
            with h5py.File(boldpath, "w") as f:
                f.create_dataset(name="bold", data=Y_bold)
            print(f"Successfully saved: {boldpath.fpath}")

        except Exception as e:
            print(f"Error processing Subject {sub_id}, Narrative {narrative}: {e}")

if __name__ == "__main__":
    from argparse import ArgumentParser

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = ArgumentParser()
    parser.add_argument(
        "-n", "--narratives", type=str, nargs="+", default=["black", "forgot"]
    )
    # ここに --subject 引数を追加
    parser.add_argument(
        "-s", "--subject", type=int, required=True, help="Subject ID (integer)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # main関数に subject_id を渡すよう変更
    main(narratives=args.narratives, subject_id=args.subject, verbose=args.verbose)
