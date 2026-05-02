# Hinglish STT

Local push-to-talk Hinglish speech-to-text for macOS.

## Run

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The app loads the local merged Whisper checkpoint at:

```text
whisper-hinglish-merged/whisper-hinglish-small-merged/model.safetensors
```

It uses Apple MPS through PyTorch when available, otherwise CPU.

## Controls

- Hold Space to record, release to transcribe.
- Or press and hold the main button.
- Saved corrections appear in `corrections/` and `/admin`.


