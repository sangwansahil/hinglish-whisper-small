class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSize = 1024;
    this.buffer = new Float32Array(this.frameSize);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const channel = input[0];
    for (let i = 0; i < channel.length; i += 1) {
      this.buffer[this.offset] = channel[i];
      this.offset += 1;
      if (this.offset === this.frameSize) {
        const out = new Int16Array(this.frameSize);
        for (let j = 0; j < this.frameSize; j += 1) {
          const sample = Math.max(-1, Math.min(1, this.buffer[j]));
          out[j] = sample < 0 ? sample * 32768 : sample * 32767;
        }
        this.port.postMessage(out.buffer, [out.buffer]);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
