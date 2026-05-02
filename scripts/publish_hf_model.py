from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = ROOT / "dist" / "hf-model"


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the prepared model folder to Hugging Face Hub.")
    parser.add_argument("repo_id", help="Hugging Face repo id, e.g. username/hinglish-whisper-small")
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--private", action="store_true", help="Create the model repository as private.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", default="Upload Hinglish Whisper checkpoint")
    args = parser.parse_args()

    if not (args.folder / "model.safetensors").exists():
        raise SystemExit(
            f"{args.folder} is not prepared. Run: python scripts/prepare_hf_repo.py --repo-id {args.repo_id}"
        )

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.folder),
        revision=args.revision,
        commit_message=args.commit_message,
    )
    print(f"Published https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
