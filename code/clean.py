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
    """
    Extract word onset and word rate regressors from alignment data.
    
    Parameters
    ----------
    narrative : str
        Narrative name (e.g., 'pieman', 'black', 'forgot')
    
    Returns
    -------
    word_onsets : np.ndarray
        Binary array indicating TRs with word onsets (shape: n_trs,)
    word_rates : np.ndarray
        Number of words per TR (shape: n_trs,)
    """
    # Load alignment CSV file
    filename = f"/disk1/MRI-Data_in-use/20_narrativefMRI/10_ds002245-v.1.0.3_Hasson/stimuli/gentle/{narrative}/align.csv"
    
    # Read CSV without header, assign column names manually
    df = pd.read_csv(filename, header=None, names=['word', 'lemma', 'onset', 'offset'])
    
    # Remove rows with missing onset values (e.g., punctuation)
    df = df.dropna(subset=['onset'])
    
    # Convert onset time (seconds) to TR number
    df['TR'] = (df['onset'] / TR[narrative]).astype(int)
    
    # Get total number of TRs for this narrative
    n_trs = TRS[narrative]
    
    # Initialize output arrays
    word_onsets = np.zeros(n_trs, dtype=np.float32)
    word_rates = np.zeros(n_trs, dtype=np.float32)
    
    # Aggregate word counts per TR
    for tr in range(n_trs):
        subdf = df[df.TR == tr]
        
        if len(subdf) > 0:
            word_onsets[tr] = 1
            word_rates[tr] = len(subdf)
    
    return word_onsets, word_rates


def get_bold(sub: int, narrative: str) -> np.ndarray:
    """
    Load and preprocess BOLD signal with confound regression.
    
    Parameters
    ----------
    sub : int
        Subject ID
    narrative : str
        Narrative name
    
    Returns
    -------
    Y_bold : np.ndarray
        Confound-regressed BOLD signal (shape: n_trs, n_vertices)
    """
    # Define path to BOLD data (left and right hemispheres)
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
    
    # Define path to confound regressors
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
    
    # Load standard confounds (motion, physiological noise, etc.)
    confounds_fn = confpath.fpath
    confounds_df, confounds_meta = load_confounds(confounds_fn)
    confounds = extract_confounds(confounds_df, confounds_meta, CONFOUND_MODEL)
    
    # Extract task-related nuisance regressors (word onsets and rates)
    word_onsets, word_rates = get_nuisance_regressors(narrative)
    task_confounds = np.zeros((len(confounds), 2))
    task_confounds[NARRATIVE_SLICE[narrative], 0] = word_onsets
    task_confounds[NARRATIVE_SLICE[narrative], 1] = word_rates
    
    # Combine all confounds
    all_confounds = np.hstack((confounds.to_numpy(), task_confounds))
    
    # Apply confound regression with detrending and standardization
    masker = subject.GiftiMasker(
        t_r=TR[narrative],
        detrend=True,
        ensure_finite=True,
        standardize="zscore_sample",
        standardize_confounds=True,
    )
    Y_bold = masker.fit_transform(paths, confounds=all_confounds)
    
    # Extract narrative time period
    Y_bold = Y_bold[NARRATIVE_SLICE[narrative]]
    
    return Y_bold


def main(narratives: str, subject_id: int, **kwargs):
    """
    Process and save confound-regressed BOLD data for specified subject and narratives.
    
    Parameters
    ----------
    narratives : list of str
        List of narrative names to process
    subject_id : int
        Subject ID to process
    **kwargs
        Additional keyword arguments (e.g., verbose)
    """
    for narrative in narratives:
        sub_id = subject_id
        
        print(f"Processing Subject: {sub_id}, Narrative: {narrative}")
        
        # Define output path
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
            # Process BOLD data
            Y_bold = get_bold(sub_id, narrative)
            print(Y_bold.shape)
            
            # Save to HDF5 file
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
        "-n", "--narratives",
        type=str,
        nargs="+",
        default=["black", "forgot"],
        help="List of narrative names to process"
    )
    parser.add_argument(
        "-s", "--subject",
        type=int,
        required=True,
        help="Subject ID (integer)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    main(narratives=args.narratives, subject_id=args.subject, verbose=args.verbose)