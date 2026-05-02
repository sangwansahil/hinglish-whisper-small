from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = ROOT / "dist" / "hf-model"


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a local or remote Hugging Face Whisper export.")
    parser.add_argument("model", nargs="?", default=str(DEFAULT_FOLDER), help="Local folder or HF repo id.")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    audio = np.zeros(16_000, dtype=np.float32)
    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    if device == "mps":
        input_features = input_features.to(dtype=torch.float16)

    with torch.inference_mode():
        ids = model.generate(input_features, max_new_tokens=8, do_sample=False)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print({"ok": True, "device": device, "text": text})


if __name__ == "__main__":
    main()
