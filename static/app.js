const statusEl = document.querySelector("#status");
const takesEl = document.querySelector("#takes");
const recordButton = document.querySelector("#recordButton");

let ws;
let audioContext;
let mediaStream;
let sourceNode;
let workletNode;
let recording = false;
let pressedBySpace = false;
let liveTake = null;
let currentAudio = [];
let orphanTimer = null;

function setStatus(text, mode = "") {
  statusEl.textContent = text;
  statusEl.dataset.mode = mode;
}

function wsURL() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

function connectWS() {
  ws = new WebSocket(wsURL());
  ws.binaryType = "arraybuffer";
  ws.addEventListener("open", () => setStatus("Ready", "ready"));
  ws.addEventListener("close", () => {
    setStatus("Reconnecting…", "warn");
    setTimeout(connectWS, 600);
  });
  ws.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "commit") commitToCard(data);
  });
}

async function ensureAudio() {
  if (audioContext && workletNode) return;
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: 16000,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  audioContext = new AudioContext({ sampleRate: 16000 });
  await audioContext.audioWorklet.addModule("/static/pcm-worklet.js");
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, "pcm-worklet");
  workletNode.port.onmessage = (event) => {
    if (!recording || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(event.data);
    currentAudio.push(new Int16Array(event.data.slice(0)));
  };
  sourceNode.connect(workletNode);
}

function makeLiveTake() {
  const node = document.createElement("article");
  node.className = "take recording";
  node.innerHTML = `
    <div class="take-row">
      <span class="take-bullet"></span>
      <div class="take-text placeholder">recording<span class="dots"></span></div>
      <button class="icon-button save" title="Save correction" type="button" disabled>Save</button>
      <span class="take-meta">live</span>
    </div>
  `;
  takesEl.prepend(node);
  return node;
}

function setLiveTranscribing(node) {
  if (!node) return;
  node.classList.remove("recording");
  node.classList.add("transcribing");
  node.querySelector(".take-text").innerHTML = 'transcribing<span class="dots"></span>';
  node.querySelector(".take-meta").textContent = "queued";
}

function commitToCard(data) {
  clearTimeout(orphanTimer);
  const node = liveTake || makeLiveTake();
  const audioChunks = currentAudio.slice();
  node.classList.remove("recording", "transcribing");
  const textEl = node.querySelector(".take-text");
  const saveButton = node.querySelector(".save");
  const metaEl = node.querySelector(".take-meta");
  const text = data.finetuned || "";
  textEl.textContent = text || "No speech detected";
  textEl.classList.toggle("placeholder", !text);
  metaEl.textContent = `${data.duration_s.toFixed(1)}s · ${data.ms}ms`;
  saveButton.disabled = !text;
  saveButton.onclick = () => saveCorrection(text, data, audioChunks);
  liveTake = null;
  currentAudio = [];
  setStatus("Ready", "ready");
}

async function saveCorrection(text, meta, audioChunks) {
  const audio = flattenPCMToFloats(audioChunks);
  const res = await fetch("/corrections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, audio, meta }),
  });
  if (!res.ok) {
    setStatus("Could not save correction", "warn");
    return;
  }
  setStatus("Correction saved", "ready");
}

function flattenPCMToFloats(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i += 1) {
      out[offset + i] = chunk[i] / 32768;
    }
    offset += chunk.length;
  }
  return out;
}

async function startRecording() {
  if (recording) return;
  await ensureAudio();
  if (audioContext.state === "suspended") await audioContext.resume();
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    setStatus("Socket not ready yet", "warn");
    return;
  }
  recording = true;
  currentAudio = [];
  liveTake = makeLiveTake();
  document.body.classList.add("is-recording");
  ws.send(JSON.stringify({ type: "reset" }));
  setStatus("Listening…", "recording");
}

function stopRecording() {
  if (!recording) return;
  recording = false;
  document.body.classList.remove("is-recording");
  setLiveTranscribing(liveTake);
  setStatus("Transcribing…", "working");
  ws.send(JSON.stringify({ type: "flush" }));
  orphanTimer = setTimeout(() => {
    if (liveTake) commitToCard({ finetuned: "", duration_s: 0, ms: 0 });
  }, 8000);
}

recordButton.addEventListener("pointerdown", startRecording);
recordButton.addEventListener("pointerup", stopRecording);
recordButton.addEventListener("pointercancel", stopRecording);
recordButton.addEventListener("pointerleave", () => {
  if (recording && !pressedBySpace) stopRecording();
});

window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.repeat || event.target.closest("input, textarea")) return;
  event.preventDefault();
  pressedBySpace = true;
  startRecording();
});

window.addEventListener("keyup", (event) => {
  if (event.code !== "Space") return;
  event.preventDefault();
  pressedBySpace = false;
  stopRecording();
});

connectWS();
