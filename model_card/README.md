---
license: other
library_name: transformers
pipeline_tag: automatic-speech-recognition
language:
  - en
  - hi
tags:
  - whisper
  - automatic-speech-recognition
  - hinglish
  - code-switching
  - romanized-hindi
base_model: openai/whisper-small
base_model_relation: finetune
---

# Hinglish Whisper Small

This is a fine-tuned Whisper Small checkpoint for Hinglish speech recognition, with an emphasis on romanized Hindi/English code-switching output.

## Intended Use

This model is intended for local or server-side automatic speech recognition of informal Hinglish speech. It is especially useful when the desired output is romanized Hinglish rather than Devanagari.

## Usage

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

repo_id = "YOUR_USERNAME/YOUR_MODEL_REPO"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

processor = AutoProcessor.from_pretrained(repo_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(repo_id, torch_dtype=dtype).to(device)
model.eval()

# audio should be a float32 mono waveform sampled at 16 kHz.
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(device)
if device == "mps":
    input_features = input_features.to(dtype=torch.float16)

with torch.inference_mode():
    predicted_ids = model.generate(input_features, max_new_tokens=192, do_sample=False)

text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(text)
```

## Local App

The companion local app lives in the source repository and provides a FastAPI/WebSocket push-to-talk recorder for macOS.

## Training Data

This model was fine-tuned on Hinglish speech examples. Add dataset details here before public release:

- dataset source and ownership
- number of hours
- train/eval split
- transcription guidelines
- whether recordings include private or sensitive speech

## Evaluation

Add quantitative evaluation before public release where possible:

| Split | WER | CER | Notes |
|---|---:|---:|---|
| TBD | TBD | TBD | Hinglish romanized output |

Local app benchmark observed fast inference on Apple Silicon, but these timings are not a substitute for ASR quality metrics.

## Limitations

- Optimized for Hinglish/code-switched speech; performance may be weaker on pure Hindi, pure English, noisy audio, accents not represented in training, or long-form speech.
- Whisper-style models are not streaming-native; low-latency streaming behavior requires additional application logic and may trade latency for stability.
- Transcriptions may contain hallucinations, omissions, or spelling variation, especially during silence or low-SNR audio.

## Safety and Privacy

Do not publish private voice recordings or transcripts without permission. If you release a training dataset, document consent, redaction, and licensing clearly.

## License

Set the final license before public release. Make sure it is compatible with the base model, training data, and any commercial use you want to allow.
