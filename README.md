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

## Notes

This recreated version uses the Hugging Face model artifacts present in this folder. The previous pywhispercpp/ggml path can be added later if you copy over a `ggml-*.bin` model.

## Publishing

This project keeps the source repo and model repo separate:

- GitHub/Git: app code, scripts, model card template, requirements.
- Hugging Face: large model weights and processor/tokenizer files.
- Local only: `venv/`, `.runtime/`, `corrections/`, `dist/`, and raw model exports in `whisper-hinglish-merged/`.

### Prepare the Hugging Face model folder

```bash
source venv/bin/activate
python scripts/prepare_hf_repo.py --repo-id YOUR_USERNAME/hinglish-whisper-small
python scripts/check_hf_export.py
```

The prepared upload folder is `dist/hf-model/`. It includes `model.safetensors`, tokenizer/processor files, `.gitattributes`, and the Hugging Face model card.

### Publish to Hugging Face

```bash
source venv/bin/activate
hf auth login
python scripts/publish_hf_model.py YOUR_USERNAME/hinglish-whisper-small --private
```

Drop `--private` when you are ready for a public release. Before public release, edit `model_card/README.md` with the real license, dataset provenance, training details, and evaluation metrics, then rerun `prepare_hf_repo.py`.

### Publish the app source to Git

```bash
scripts/init_git_repo.sh
git commit -m "Initial local Hinglish STT app"
git branch -M main
git remote add origin YOUR_GIT_REMOTE_URL
git push -u origin main
```

The large model files are intentionally ignored by Git. If you later want a Git repo that contains weights directly, install Git LFS first and make that a separate, deliberate choice.
