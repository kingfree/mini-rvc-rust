class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.outputBuffer = [];
    this.inputGain = 1.0;
    this.outputGain = 1.0;
    this.threshold = -60; // dB
  }

  static get parameterDescriptors() {
    return [
      { name: 'inputGain', defaultValue: 1.0, minValue: 0.0, maxValue: 10.0 },
      { name: 'outputGain', defaultValue: 1.0, minValue: 0.0, maxValue: 10.0 },
      { name: 'threshold', defaultValue: -60.0, minValue: -100.0, maxValue: 0.0 },
    ];
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    
    // Update parameters
    this.inputGain = parameters.inputGain.length > 1 ? parameters.inputGain[0] : parameters.inputGain[0];
    this.outputGain = parameters.outputGain.length > 1 ? parameters.outputGain[0] : parameters.outputGain[0];
    const thresholdDB = parameters.threshold.length > 1 ? parameters.threshold[0] : parameters.threshold[0];
    const thresholdLin = Math.pow(10, thresholdDB / 20);

    // 1. Handle Input
    if (input.length > 0 && input[0].length > 0) {
      const inputData = input[0];
      const processedInput = new Float32Array(inputData.length);
      
      let maxAmp = 0.0;
      for (let i = 0; i < inputData.length; i++) {
        let sample = inputData[i] * this.inputGain;
        if (Math.abs(sample) > maxAmp) maxAmp = Math.abs(sample);
        processedInput[i] = sample;
      }

      // Simple Noise Gate
      if (maxAmp >= thresholdLin) {
        this.port.postMessage({
          type: 'audio-input',
          samples: processedInput
        });
      } else {
        // Send silence or skip? 
        // If we skip, backend ring buffer might starve or drift. 
        // Better to send silence to keep timing, or backend handles silence.
        // For simplicity in this demo, we assume continuous stream required.
        this.port.postMessage({
          type: 'audio-input',
          samples: processedInput // sending silence (or near silence)
        });
      }
    }

    // 2. Handle Output
    if (output.length > 0 && output[0].length > 0) {
      const outputData = output[0];
      
      if (this.outputBuffer.length >= outputData.length) {
        const chunk = this.outputBuffer.splice(0, outputData.length);
        
        for (let i = 0; i < outputData.length; i++) {
          outputData[i] = chunk[i] * this.outputGain;
        }
        
        // Copy to other channels
        for (let channel = 1; channel < output.length; channel++) {
          output[channel].set(outputData);
        }
      } else {
        outputData.fill(0);
      }
    }

    // Handle messages from main thread
    this.port.onmessage = (event) => {
      if (event.data.type === 'audio-output') {
        this.outputBuffer.push(...event.data.samples);
        if (this.outputBuffer.length > 48000 * 2) {
          this.outputBuffer.splice(0, this.outputBuffer.length - 48000);
        }
      }
    };

    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);