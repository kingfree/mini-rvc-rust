class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.outputBuffer = [];
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    // 1. Send input to main thread
    if (input.length > 0 && input[0].length > 0) {
      // We only take the first channel for simplicity (RVC is mono)
      const inputData = input[0];
      this.port.postMessage({
        type: 'audio-input',
        samples: inputData
      });
    }

    // 2. Handle output from buffer
    if (output.length > 0 && output[0].length > 0) {
      const outputData = output[0];
      
      if (this.outputBuffer.length >= outputData.length) {
        const chunk = this.outputBuffer.splice(0, outputData.length);
        outputData.set(chunk);
        
        // If stereo, copy to other channels
        for (let i = 1; i < output.length; i++) {
          output[i].set(chunk);
        }
      } else {
        // Not enough data, output silence
        outputData.fill(0);
      }
    }

    // Handle messages from main thread
    this.port.onmessage = (event) => {
      if (event.data.type === 'audio-output') {
        this.outputBuffer.push(...event.data.samples);
        
        // Cap the buffer size to avoid massive lag
        if (this.outputBuffer.length > 48000 * 2) {
          this.outputBuffer.splice(0, this.outputBuffer.length - 48000);
        }
      }
    };

    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);
