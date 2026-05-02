from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
CORRECTIONS_DIR = ROOT / "corrections"
MANIFEST_PATH = CORRECTIONS_DIR / "manifest.jsonl"

MODEL_DIR = ROOT / "whisper-hinglish-merged" / "whisper-hinglish-small-merged"
PROCESSOR_DIR = ROOT / "whisper-hinglish-merged"
RUNTIME_DIR = ROOT / ".runtime"
RUNTIME_PROCESSOR_DIR = RUNTIME_DIR / "processor"

SAMPLE_RATE = 16_000
MIN_AUDIO_SECONDS = 0.2
MAX_WINDOW_SECS = 25.0
RMS_THRESHOLD = 0.005

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOG = logging.getLogger("hinglish-stt")

EXECUTOR = ThreadPoolExecutor(max_workers=1)


class LocalWhisper:
    def __init__(self) -> None:
        self.model_dir = MODEL_DIR
        self.processor_dir = PROCESSOR_DIR
        self.ready = False
        self.device = "cpu"
        self.dtype_name = "float32"
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None

    def load(self) -> None:
        if self.ready:
            return
        if not (self.model_dir / "model.safetensors").exists():
            raise RuntimeError(f"Missing model file: {self.model_dir / 'model.safetensors'}")

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        if torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
            self.dtype_name = "float16"
        else:
            self.device = "cpu"
            dtype = torch.float32
            self.dtype_name = "float32"

        processor_dir = prepare_processor_dir(self.processor_dir)
        LOG.info("loading local Whisper model from %s", self.model_dir)
        self.processor = AutoProcessor.from_pretrained(processor_dir, local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.ready = True
        LOG.info("model ready on %s (%s)", self.device, self.dtype_name)

        # Warm the graph once so the first real utterance is not punished.
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self.transcribe(silence)

    def transcribe(self, audio: np.ndarray) -> str:
        self.load()
        assert self.processor is not None
        assert self.model is not None
        assert self.torch is not None

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        if self.device == "mps":
            input_features = input_features.to(dtype=self.torch.float16)

        with self.torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                max_new_tokens=192,
                do_sample=False,
            )
        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return clean_text(text)


TRANSCRIBER = LocalWhisper()


def prepare_processor_dir(source_dir: Path) -> Path:
    RUNTIME_PROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        if source.is_file() and source.suffix == ".json":
            target = RUNTIME_PROCESSOR_DIR / source.name
            data = json.loads(source.read_text(encoding="utf-8"))
            if source.name == "tokenizer_config.json" and isinstance(data.get("extra_special_tokens"), list):
                data.pop("extra_special_tokens", None)
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif source.is_file() and source.name in {"vocab.json", "merges.txt"}:
            target = RUNTIME_PROCESSOR_DIR / source.name
            target.write_bytes(source.read_bytes())
    return RUNTIME_PROCESSOR_DIR


def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def pcm_bytes_to_float32(payload: bytes) -> np.ndarray:
    if len(payload) < 2:
        return np.empty(0, dtype=np.float32)
    pcm = np.frombuffer(payload, dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(audio)))))


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm16.tobytes())


async def run_transcription(audio: np.ndarray) -> tuple[str, int]:
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    text = await loop.run_in_executor(EXECUTOR, TRANSCRIBER.transcribe, audio)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return text, elapsed_ms


@asynccontextmanager
async def lifespan(_: FastAPI):
    CORRECTIONS_DIR.mkdir(exist_ok=True)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(EXECUTOR, TRANSCRIBER.load)
    yield


app = FastAPI(title="Hinglish STT", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
async def admin() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": str(MODEL_DIR.relative_to(ROOT)),
        "processor": str(PROCESSOR_DIR.relative_to(ROOT)),
        "device": TRANSCRIBER.device,
        "ready": TRANSCRIBER.ready,
    }


@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    chunks: list[np.ndarray] = []
    take_id = str(uuid.uuid4())

    async def flush() -> None:
        nonlocal chunks, take_id
        if chunks:
            audio = np.concatenate(chunks)
        else:
            audio = np.empty(0, dtype=np.float32)
        chunks = []

        max_samples = int(MAX_WINDOW_SECS * SAMPLE_RATE)
        if audio.size > max_samples:
            audio = audio[-max_samples:]

        duration_s = audio.size / SAMPLE_RATE
        level = rms(audio)
        current_take_id = take_id
        take_id = str(uuid.uuid4())

        if duration_s < MIN_AUDIO_SECONDS or level < RMS_THRESHOLD:
            await ws.send_json(
                {
                    "type": "commit",
                    "take_id": current_take_id,
                    "finetuned": "",
                    "vanilla": "",
                    "duration_s": round(duration_s, 2),
                    "ms": 0,
                    "silenced": True,
                }
            )
            return

        text, elapsed_ms = await run_transcription(audio)
        rtf = elapsed_ms / 1000 / max(duration_s, 0.001)
        LOG.info(
            "transcribed %sms | %.1fs audio | RTF %.2f | %r",
            elapsed_ms,
            duration_s,
            rtf,
            text,
        )
        await ws.send_json(
            {
                "type": "commit",
                "take_id": current_take_id,
                "finetuned": text,
                "vanilla": "",
                "duration_s": round(duration_s, 2),
                "ms": elapsed_ms,
            }
        )

    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                chunk = pcm_bytes_to_float32(msg["bytes"])
                if chunk.size:
                    chunks.append(chunk)
                continue

            text = msg.get("text")
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            msg_type = data.get("type")
            if msg_type == "reset":
                chunks = []
                take_id = str(uuid.uuid4())
            elif msg_type == "flush":
                await flush()
    except WebSocketDisconnect:
        return


@app.get("/corrections")
async def list_corrections() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_dir in sorted(CORRECTIONS_DIR.iterdir(), reverse=True):
        if not sample_dir.is_dir():
            continue
        text_path = sample_dir / "text.txt"
        if not text_path.exists():
            continue
        meta_path = sample_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "id": sample_dir.name,
                "text": clean_text(text_path.read_text(encoding="utf-8")),
                "created_at": datetime.fromtimestamp(sample_dir.stat().st_mtime, timezone.utc).isoformat(),
                "has_audio": (sample_dir / "audio.wav").exists(),
                "meta": meta,
            }
        )
    return rows


@app.post("/corrections")
async def create_correction(payload: dict[str, Any]) -> JSONResponse:
    text = clean_text(str(payload.get("text", "")))
    audio = payload.get("audio")
    meta = payload.get("meta", {})
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    sample_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    sample_dir = CORRECTIONS_DIR / sample_id
    sample_dir.mkdir(parents=True, exist_ok=False)
    (sample_dir / "text.txt").write_text(text + "\n", encoding="utf-8")
    (sample_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if isinstance(audio, list):
        arr = np.asarray(audio, dtype=np.float32)
        write_wav(sample_dir / "audio.wav", arr)

    row = {
        "id": sample_id,
        "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "has_audio": (sample_dir / "audio.wav").exists(),
        "meta": meta,
    }
    with MANIFEST_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return JSONResponse(row)


@app.patch("/corrections/{sample_id}")
async def update_correction(sample_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    sample_dir = CORRECTIONS_DIR / sample_id
    text_path = sample_dir / "text.txt"
    if not text_path.exists():
        raise HTTPException(status_code=404, detail="correction not found")
    text = clean_text(str(payload.get("text", "")))
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    text_path.write_text(text + "\n", encoding="utf-8")
    return {"ok": True, "id": sample_id, "text": text}


@app.delete("/corrections/{sample_id}")
async def delete_correction(sample_id: str) -> dict[str, Any]:
    sample_dir = CORRECTIONS_DIR / sample_id
    if not sample_dir.exists():
        raise HTTPException(status_code=404, detail="correction not found")
    for child in sample_dir.iterdir():
        child.unlink()
    sample_dir.rmdir()
    return {"ok": True, "id": sample_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
