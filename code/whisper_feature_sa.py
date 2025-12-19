"""Generate embedding for a word-aligned transcript.
Modified to use sentence_alignments.csv for sentence-level processing.
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
    "whisper-tiny": "openai/whisper-tiny.en", # 4 layers
    "whisper-medium": "openai/whisper-medium.en", # 8 layers
    # large has 32 layers with model_d = 1280
    "whisper-large": "openai/whisper-large-v3",
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
        root="/home/s-kawashima/research/output/features",
        datatype=modelname + "-audio-sentence",
        suffix=None,
        ext="pkl",
    )

    sfreq = 16000
    audio_fpattern = "/disk1/MRI-Data_in-use/20_narrativefMRI/10_ds002245-v.1.0.3_Hasson/stimuli/{}_audio.wav"
    sentence_csv_pattern = "/home/s-kawashima/research/output/sentence/{}/sentence_alignments.csv"
    
    for narrative in narratives:
        print(f"Processing narrative: {narrative}")

        # Load sentence alignments CSV for this narrative
        sentence_csv = sentence_csv_pattern.format(narrative)
        print(f"Loading sentence alignments from {sentence_csv}...")
        sentence_df = pd.read_csv(sentence_csv)

        # Load stimuli
        audio = load_audio(audio_fpattern.format(narrative))
        
        # Group by sentence_id
        grouped = sentence_df.groupby('sentence_id')
        
        # Setup examples (one per word, but using sentence-level audio context)
        examples = []
        word_data = []
        
        for sentence_id, sentence_group in tqdm(grouped, desc="Building"):
            # Get sentence boundaries
            sentence_start = sentence_group['sentence_start'].iloc[0]
            sentence_end = sentence_group['absolute_end'].max()
            
            # Extract audio for the entire sentence
            start_sample = int(sentence_start * sfreq)
            end_sample = int(sentence_end * sfreq)
            sentence_audio = audio[start_sample:end_sample]
            
            # Get all words in this sentence
            sentence_words = sentence_group.sort_values('absolute_start')
            sentence_text = " " + " ".join(sentence_words['word'].tolist())
            
            # Tokenize the sentence text
            sentence_tokens = tokenizer.encode(sentence_text, return_tensors="pt")
            
            # Process input features for the sentence audio
            input_features = feature_extractor(
                sentence_audio,
                sampling_rate=sfreq,
                return_attention_mask=True,
                return_tensors="pt",
            )
            
            # For each word in the sentence, create an example
            for idx, (_, row) in enumerate(sentence_words.iterrows()):
                word_duration = row['absolute_end'] - row['absolute_start']
                word_text = row['word']
                n_tokens = len(tokenizer.tokenize(" " + word_text))
                
                examples.append(
                    dict(
                        sentence_id=sentence_id,
                        word=word_text,
                        word_idx_in_sentence=idx,
                        duration=word_duration,
                        absolute_start=row['absolute_start'],
                        absolute_end=row['absolute_end'],
                        sentence_audio=sentence_audio,
                        sentence_audio_dur=sentence_audio.size / sfreq,
                        input_features=input_features.input_features,
                        audio_samples=input_features.attention_mask[0].nonzero()[-1].item() + 1,
                        decoder_input_ids=sentence_tokens,
                        n_tokens=n_tokens,
                        n_words_in_sentence=len(sentence_words),
                    )
                )
                
                word_data.append({
                    'sentence_id': sentence_id,
                    'word': word_text,
                    'absolute_start': row['absolute_start'],
                    'absolute_end': row['absolute_end'],
                    'duration': word_duration,
                })

        # Convert word_data to DataFrame for saving
        df = pd.DataFrame(word_data)

        # Run through model
        con_embeddings = []
        enc_embeddings = []
        dec_embeddings = []
        with torch.no_grad():
            for example in tqdm(examples, desc="Extracting"):
                # Duration of word by 20 ms
                n_frames = int(np.ceil(example["duration"] / 0.2))
                # / 2 to account for conv from 3000 -> 1500 frames
                end_frame = int(np.ceil(example["audio_samples"] / 2))
                
                # Calculate temporal slice for this word within the sentence
                # Word position relative to sentence start
                word_rel_start = example['absolute_start'] - (example['absolute_end'] - example['duration'])
                word_frame_start = int(np.ceil((word_rel_start * sfreq / 2)))
                word_frame_end = word_frame_start + n_frames
                
                # Use the last frames corresponding to the word's temporal position
                temporal_slice = slice(max(0, word_frame_end - n_frames), word_frame_end)
                
                # Decoder slice for this specific word's tokens
                decoder_emb_slice = slice(-(example["n_tokens"] + 1), -1)

                # Forward pass through model
                outputs = model(
                    input_features=example["input_features"].to(device),
                    decoder_input_ids=example["decoder_input_ids"].to(device),
                    output_hidden_states=True,
                )

                # Extract activations
                conv_state = outputs["encoder_hidden_states"][0]
                encoder_state = outputs["encoder_hidden_states"][-1]
                decoder_states = outputs["decoder_hidden_states"]

                # Extract portion of state for each example(/word)
                conv_state = conv_state[0, temporal_slice].mean(0)
                encoder_state = encoder_state[0, temporal_slice].mean(0)
                decoder_states = torch.stack(decoder_states)
                decoder_states = decoder_states[:, 0, decoder_emb_slice].mean(1)

                con_embeddings.append(conv_state.numpy(force=True))
                enc_embeddings.append(encoder_state.numpy(force=True))
                dec_embeddings.append(decoder_states.numpy(force=True))

        # Save transcript
        epath.update(narrative=narrative, ext="pkl")
        epath.mkdirs()
        df.to_pickle(epath)

        con_embeddings = np.stack(con_embeddings)
        enc_embeddings = np.stack(enc_embeddings)
        dec_embeddings = np.stack(dec_embeddings)

        # Save embeddings
        epath.update(narrative=narrative, ext=".h5")
        with h5py.File(epath, "w") as f:
            f.create_dataset(name="activations_conv", data=con_embeddings)
            f.create_dataset(name="activations_enc", data=enc_embeddings)
            f.create_dataset(name="activations_dec", data=dec_embeddings)
        
        print(f"Saved embeddings for {narrative}: {len(examples)} words across {len(grouped)} sentences")


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