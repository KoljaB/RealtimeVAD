// Forwards every 128-sample render quantum (16 kHz context) to the page.
// Without a connected input it sends silence, so the stream clock never stops.
class Capture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    this.port.postMessage(channel ? channel.slice() : new Float32Array(128));
    return true;
  }
}
registerProcessor("capture", Capture);
