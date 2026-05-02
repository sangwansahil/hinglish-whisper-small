from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "whisper-hinglish-merged" / "whisper-hinglish-small-merged"
DEFAULT_PROCESSOR_DIR = ROOT / "whisper-hinglish-merged"
DEFAULT_CARD = ROOT / "model_card" / "README.md"
DEFAULT_OUT = ROOT / "dist" / "hf-model"

PROCESSOR_FILES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}

MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
}


def copy_json_sanitized(source: Path, target: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    if source.name == "tokenizer_config.json" and isinstance(data.get("extra_special_tokens"), list):
        data.pop("extra_special_tokens", None)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        copy_json_sanitized(source, target)
    else:
        shutil.copy2(source, target)


def prepare(model_dir: Path, processor_dir: Path, out_dir: Path, model_card: Path, repo_id: str | None) -> None:
    if not (model_dir / "model.safetensors").exists():
        raise SystemExit(f"Missing model.safetensors in {model_dir}")
    if not processor_dir.exists():
        raise SystemExit(f"Missing processor directory: {processor_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for name in PROCESSOR_FILES:
        source = processor_dir / name
        if source.exists():
            copy_file(source, out_dir / name)

    for name in MODEL_FILES:
        source = model_dir / name
        if source.exists():
            copy_file(source, out_dir / name)

    readme = model_card.read_text(encoding="utf-8")
    if repo_id:
        readme = readme.replace("YOUR_USERNAME/YOUR_MODEL_REPO", repo_id)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    (out_dir / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.bin filter=lfs diff=lfs merge=lfs -text\n"
        "*.gguf filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )

    print(f"Prepared Hugging Face model repo at {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a clean Hugging Face model repository folder.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--processor-dir", type=Path, default=DEFAULT_PROCESSOR_DIR)
    parser.add_argument("--model-card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repo-id", help="Optional HF repo id, e.g. username/hinglish-whisper-small")
    args = parser.parse_args()
    prepare(args.model_dir, args.processor_dir, args.out_dir, args.model_card, args.repo_id)


if __name__ == "__main__":
    main()
