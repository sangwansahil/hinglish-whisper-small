#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .git ]; then
  git init
fi

git add \
  .gitignore \
  .gitattributes \
  README.md \
  requirements.txt \
  server.py \
  static \
  scripts \
  model_card

git status --short

cat <<'MSG'

Next:
  git commit -m "Initial local Hinglish STT app"
  git branch -M main
  git remote add origin <YOUR_GIT_REMOTE_URL>
  git push -u origin main

Large model files are intentionally ignored in this source repo.
Publish them to Hugging Face with scripts/publish_hf_model.py.
MSG
