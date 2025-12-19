"""Generate embedding for a word-aligned transcript.
"""

import json
import subprocess

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel, WhisperTokenizer
from util.path import Path

# short names for long model names
HFMODELS = {
    "whisper-tiny": "openai/whisper-tiny.en",
    "whisper-medium": "openai/whisper-medium.en",
    # large has 32 layers with model_d = 1280
    "whisper-large": "openai/whisper-large-v3",
    # "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2",
}


def load_audio(file: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def main(narratives: list[str], modelname: str, device: str):

    hfmodelname = HFMODELS[modelname]

    # Load model
    print("Loading model...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(hfmodelname)
    tokenizer = WhisperTokenizer.from_pretrained(
        hfmodelname, task="transcribe", language="english"
    )
    model = WhisperModel.from_pretrained(hfmodelname)

    print(
        f"Model : {hfmodelname} ({modelname})"
        f"\nLayers (encoder): {model.config.encoder_layers}"
        f"\nLayers (decoder): {model.config.decoder_layers}"
        f"\nEmbDim: {model.config.d_model}"
        f"\nCxtLen: {model.config.max_length}"
        f"\nDevice: {device}"
    )
    model = model.eval()
    model = model.to(device)

    epath = Path(
        root="/home/s-kawashima/research/output/features", #出力ディレクトリ
        datatype=modelname,
        suffix=None,
        ext="pkl",
    )

    sfreq = 16000
    audio_fpattern = "/disk1/MRI-Data_in-use/20_narrativefMRI/10_ds002245-v.1.0.3_Hasson/stimuli/{}_audio.wav" #音声ファイルパスパターン
    transcript_fpattern = "/disk1/MRI-Data_in-use/20_narrativefMRI/10_ds002245-v.1.0.3_Hasson/stimuli/gentle/{}/align.json" #トランスクリプトファイルパスパターン
    for narrative in narratives:
        print(narrative)

        # Load stimuli
        audio = load_audio(audio_fpattern.format(narrative))
        transcript_file = transcript_fpattern.format(narrative)
        with open(transcript_file, "r") as f:
            d = json.load(f)
        
        # --- JSON形式に応じて処理を分ける ---
        if "word_segments" in d:
            words = d["word_segments"]
        
        elif "words" in d:  # WhisperX align.json形式
            words = [
                {
                    "word": w.get("word", w.get("alignedWord", "")),
                    "start": w.get("start", None),
                    "end": w.get("end", None),
                }
                for w in d["words"]
                if w.get("start") is not None and w.get("end") is not None
            ]
        
        elif "segments" in d:
            # WhisperX transcript.json 形式対応
            words = []
            for s in d["segments"]:
                for w in s.get("words", []):
                    words.append(
                        {
                            "word": w.get("word", w.get("alignedWord", "")),
                            "start": w.get("start", s.get("start")),
                            "end": w.get("end", s.get("end")),
                        }
                    )
        else:
            raise KeyError("No word-level information found in transcript JSON")
        
        df = pd.DataFrame(words)
        df["start"] = df["start"].ffill()
        df["end"] = df["end"].ffill()

        # Setup examples
        examples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building"):
            end_s = row.end
            start_s = max(0, end_s - 30)
            sub_df = df[(df.start > start_s) & (df.start < end_s)]

            sub_audio = audio[int(start_s * sfreq) : int(end_s * sfreq)]

            input_features = feature_extractor(
                sub_audio,
                sampling_rate=sfreq,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text = " " + " ".join(sub_df.word.tolist())
            tokens = tokenizer.encode(text, return_tensors="pt")

            examples.append(
                dict(
                    word=row.word,
                    duration=row.end - row.start,
                    audio=sub_audio,
                    audio_dur=sub_audio.size / sfreq,
                    input_features=input_features.input_features,
                    audio_samples=input_features.attention_mask[0].nonzero()[-1].item()
                    + 1,
                    decoder_input_ids=tokens,
                    n_tokens=len(tokenizer.tokenize(" " + row.word)),
                )
            )

        # Run through model
        con_embeddings = []
        enc_embeddings = []
        dec_embeddings = []
        with torch.no_grad():
            for example in tqdm(examples, desc="Extracting"):
                # duration of word by 20 ms
                n_frames = int(np.ceil(example["duration"] / 0.2))
                # / 2 to account for conv from 3000 -> 1500 frames
                end_frame = int(np.ceil(example["audio_samples"] / 2))
                temporal_slice = slice(end_frame - n_frames, end_frame + 1)
                decoder_emb_slice = slice(-(example["n_tokens"] + 1), -1)

                # forward pass thru model
                outputs = model(
                    input_features=example["input_features"].to(device),
                    decoder_input_ids=example["decoder_input_ids"].to(device),
                    output_hidden_states=True,
                )

                # extract activations
                conv_state = outputs["encoder_hidden_states"][0]
                encoder_state = outputs["encoder_hidden_states"][-1]
                decoder_states = outputs["decoder_hidden_states"]

                # extract portion of state for each example(/word)
                conv_state = conv_state[0, temporal_slice].mean(0)
                encoder_state = encoder_state[0, temporal_slice].mean(0)
                decoder_states = torch.stack(decoder_states)
                decoder_states = decoder_states[:, 0, decoder_emb_slice].mean(1)

                con_embeddings.append(conv_state.numpy(force=True))
                enc_embeddings.append(encoder_state.numpy(force=True))
                dec_embeddings.append(decoder_states.numpy(force=True))

        # save transcript
        epath.update(narrative=narrative, ext="pkl")
        epath.mkdirs()
        df.to_pickle(epath)

        con_embeddings = np.stack(con_embeddings)
        enc_embeddings = np.stack(enc_embeddings)
        dec_embeddings = np.stack(dec_embeddings)

        # save embeddings
        epath.update(narrative=narrative, ext=".h5")
        with h5py.File(epath, "w") as f:
            f.create_dataset(name="activations_conv", data=con_embeddings)
            f.create_dataset(name="activations_enc", data=enc_embeddings)
            f.create_dataset(name="activations_dec", data=dec_embeddings)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--modelname", default="whisper-tiny")
    parser.add_argument(
        "-n", "--narratives", type=str, nargs="+", default=["black", "forgot"]
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    main(**vars(parser.parse_args()))
