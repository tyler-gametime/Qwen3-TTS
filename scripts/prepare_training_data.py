#!/usr/bin/env python3
"""Prepare training data for MTP validation using Qwen3-TTS-Tokenizer-12Hz.

This script:
1. Loads wav files from a directory
2. Extracts audio codes using the Qwen3-TTS tokenizer
3. Creates a JSONL file suitable for sft_12hz.py training

Usage:
    python scripts/prepare_training_data.py \
        --wav_dir /path/to/wavs \
        --output_jsonl /path/to/output.jsonl \
        --transcripts_file /path/to/transcripts.txt  # optional
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio


def load_audio(wav_path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load and resample audio to target sample rate."""
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for MTP validation")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing wav files")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--transcripts_file", type=str, help="Optional file with transcripts (wav_name|transcript)")
    parser.add_argument("--codec_model", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--max_files", type=int, default=10, help="Max number of files to process")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load transcripts if provided
    transcripts = {}
    if args.transcripts_file and os.path.exists(args.transcripts_file):
        with open(args.transcripts_file) as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    name, text = line.split("|", 1)
                    transcripts[name] = text
        print(f"Loaded {len(transcripts)} transcripts")

    # Load codec tokenizer
    print(f"Loading codec model: {args.codec_model}")
    from qwen_tts.core.models.codec import Qwen3AudioTokenizer

    tokenizer = Qwen3AudioTokenizer.from_pretrained(args.codec_model)
    tokenizer.model = tokenizer.model.to(device)
    tokenizer.model.eval()
    print("Codec model loaded successfully")

    # Find wav files
    wav_dir = Path(args.wav_dir)
    wav_files = list(wav_dir.glob("*.wav"))[:args.max_files]
    print(f"Found {len(wav_files)} wav files (processing up to {args.max_files})")

    if not wav_files:
        print("No wav files found!")
        return

    # Process each wav file
    samples = []
    for i, wav_path in enumerate(wav_files):
        print(f"Processing {i+1}/{len(wav_files)}: {wav_path.name}")
        try:
            # Load audio
            waveform = load_audio(str(wav_path), target_sr=24000)

            # Move to device and encode
            waveform = waveform.to(device)

            with torch.no_grad():
                # The tokenizer expects audio on the same device as the model
                codes = tokenizer.encode(waveform)

            # Get transcript
            transcript = transcripts.get(wav_path.stem, f"Sample text for {wav_path.stem}")

            # Create sample entry
            sample = {
                "audio_path": str(wav_path.absolute()),
                "text": transcript,
                "codes": codes.cpu().tolist() if isinstance(codes, torch.Tensor) else codes,
                "duration": waveform.shape[-1] / 24000,
            }
            samples.append(sample)
            print(f"  Duration: {sample['duration']:.2f}s, Codes shape: {codes.shape if isinstance(codes, torch.Tensor) else 'N/A'}")

        except Exception as e:
            print(f"  Error processing {wav_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Write JSONL
    print(f"\nWriting {len(samples)} samples to {args.output_jsonl}")
    with open(args.output_jsonl, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print("Done!")


if __name__ == "__main__":
    main()
