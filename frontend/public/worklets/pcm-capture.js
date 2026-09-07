/**
 * PCM capture worklet.
 *
 * Replaces the deprecated ScriptProcessorNode. Pulls each 128-frame block of
 * mono float32 audio off the input and posts a copy to the main thread, which
 * converts it to int16 PCM and streams it to the forensic backend.
 *
 * Loaded by frontend/app/page.tsx via
 * `audioContext.audioWorklet.addModule("/worklets/pcm-capture.js")`.
 */
class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];

    if (input && input.length > 0 && input[0]) {
      // Copy: the buffer is reused by the audio thread after process() returns.
      this.port.postMessage(new Float32Array(input[0]));
    }

    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
