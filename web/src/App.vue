<template>
  <div class="mashutan-app">
    <header>
      <h1>🔮 Mashutan RVC 占卜面板</h1>
      <p class="subtitle">让你的声音在 Rust 的魔力中焕发新生 ✨</p>
    </header>

    <main>
      <section class="card realtime-section">
        <h2>🌀 实时魔法 (WebSocket)</h2>
        
        <div class="visualizers">
          <div class="viz-container">
            <span class="viz-label">🎤 输入 (Input)</span>
            <canvas ref="inputCanvasRef" width="300" height="100"></canvas>
          </div>
          <div class="viz-container">
            <span class="viz-label">🔊 输出 (Output)</span>
            <canvas ref="outputCanvasRef" width="300" height="100"></canvas>
          </div>
        </div>

        <div class="controls">
          <div class="control-item">
            <label>选择模型 (Model)</label>
            <select v-model="selectedModel" @change="changeModel" class="model-select">
              <option v-for="model in models" :key="model.id" :value="model.id">
                {{ model.name }}
              </option>
            </select>
          </div>

          <div class="control-item">
            <label>音高偏移 (Pitch): {{ pitchShift }}</label>
            <input 
              type="range" 
              min="-12" 
              max="12" 
              step="1" 
              v-model="pitchShift"
              @input="sendPitchShift"
            >
          </div>
          <button 
            :class="['btn-realtime', isRealtime ? 'active' : '']"
            @click="toggleRealtime"
          >
            {{ isRealtime ? '停止咏唱' : '开启实时变换' }}
          </button>
        </div>
        <div v-if="isRealtime" class="status">
          <span class="pulse"></span> 魔法持续生效中... <span style="margin-left: 10px; font-size: 0.9em; color: #666;">延迟: {{ latency }} ms</span>
        </div>
      </section>
    </main>

    <footer>
      <p>由 玛修坦 & 森亚露露卡 强力驱动 | Powered by Candle & Rust 🦀</p>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

// Real-time state
const isRealtime = ref(false)
const pitchShift = ref(0)
const models = ref<{id: string, name: string}[]>([])
const selectedModel = ref<string>('')
const latency = ref(0)

let audioContext: AudioContext | null = null
let ws: WebSocket | null = null
let processorNode: AudioWorkletNode | null = null
let source: MediaStreamAudioSourceNode | null = null

// Visualization Refs
const inputCanvasRef = ref<HTMLCanvasElement | null>(null)
const outputCanvasRef = ref<HTMLCanvasElement | null>(null)
let inputAnalyser: AnalyserNode | null = null
let outputAnalyser: AnalyserNode | null = null
let animationId: number | null = null

onMounted(async () => {
  try {
    const res = await fetch('/api/models')
    const data = await res.json()
    models.value = data
    if (models.value.length > 0) {
      selectedModel.value = models.value[0].id
    }
  } catch (e) {
    console.error("Failed to load models", e)
  }
})

const changeModel = () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ 
      pitch_shift: Number(pitchShift.value),
      model_id: selectedModel.value 
    }))
  }
}

const sendPitchShift = () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ 
      pitch_shift: Number(pitchShift.value),
      model_id: selectedModel.value
    }))
  }
}

const drawVisualizers = () => {
  if (!isRealtime.value) return

  const draw = (analyser: AnalyserNode | null, canvas: HTMLCanvasElement | null, color: string) => {
    if (!analyser || !canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    analyser.getByteTimeDomainData(dataArray)

    ctx.fillStyle = '#f8f9fa' // Slightly off-white background
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.lineWidth = 2
    ctx.strokeStyle = color
    ctx.beginPath()

    const sliceWidth = canvas.width * 1.0 / bufferLength
    let x = 0

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0
      const y = v * canvas.height / 2

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }

      x += sliceWidth
    }

    ctx.lineTo(canvas.width, canvas.height / 2)
    ctx.stroke()
  }

  draw(inputAnalyser, inputCanvasRef.value, '#6c5ce7')
  draw(outputAnalyser, outputCanvasRef.value, '#00b894')

  animationId = requestAnimationFrame(drawVisualizers)
}

const toggleRealtime = async () => {
  if (isRealtime.value) {
    stopRealtime()
    return
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    audioContext = new AudioContext({ sampleRate: 16000 })
    
    await audioContext.audioWorklet.addModule('/audio-processor.js')
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws`
    ws = new WebSocket(wsUrl)
    ws.binaryType = 'arraybuffer'

    processorNode = new AudioWorkletNode(audioContext, 'audio-processor')
    source = audioContext.createMediaStreamSource(stream)
    
    // Setup Visualizers
    inputAnalyser = audioContext.createAnalyser()
    outputAnalyser = audioContext.createAnalyser()
    inputAnalyser.fftSize = 256
    outputAnalyser.fftSize = 256

    // Graph: Source -> InputAnalyser -> Processor -> OutputAnalyser -> Destination
    source.connect(inputAnalyser)
    inputAnalyser.connect(processorNode)
    processorNode.connect(outputAnalyser)
    outputAnalyser.connect(audioContext.destination)

    processorNode.port.onmessage = (event) => {
      if (event.data.type === 'audio-input' && ws?.readyState === WebSocket.OPEN) {
        const samples = event.data.samples
        ws.send(samples.buffer)
      }
    }

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const samples = new Float32Array(event.data)
        processorNode?.port.postMessage({
          type: 'audio-output',
          samples: samples
        })
      } else if (typeof event.data === 'string') {
        try {
          const meta = JSON.parse(event.data)
          if (meta.latency !== undefined) {
            latency.value = meta.latency
          }
        } catch (e) {
          // ignore
        }
      }
    }

    ws.onopen = () => {
      isRealtime.value = true
      changeModel() 
      drawVisualizers()
    }

    ws.onclose = () => {
      stopRealtime()
    }

  } catch (e) {
    console.error(e)
    alert('无法开启实时魔法: ' + e)
    stopRealtime()
  }
}

const stopRealtime = () => {
  isRealtime.value = false
  if (animationId) {
    cancelAnimationFrame(animationId)
    animationId = null
  }
  
  ws?.close()
  ws = null
  source?.disconnect()
  source = null
  processorNode?.disconnect()
  processorNode = null
  inputAnalyser?.disconnect()
  inputAnalyser = null
  outputAnalyser?.disconnect()
  outputAnalyser = null
  audioContext?.close()
  audioContext = null
}

onUnmounted(() => {
  stopRealtime()
})
</script>

<style scoped>
.mashutan-app {
  max-width: 800px;
  margin: 0 auto;
  padding: 40px 20px;
  font-family: 'Inter', -apple-system, sans-serif;
  color: #2d3436;
}

header {
  text-align: center;
  margin-bottom: 40px;
}

h1 {
  font-size: 2.5rem;
  background: linear-gradient(135deg, #6c5ce7, #a29bfe);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 10px;
}

.subtitle {
  color: #636e72;
  font-style: italic;
}

.card {
  background: white;
  border-radius: 20px;
  padding: 30px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
  margin-bottom: 20px;
  transition: transform 0.3s ease;
}

.card:hover {
  transform: translateY(-5px);
}

h2 {
  font-size: 1.5rem;
  margin-top: 0;
  margin-bottom: 20px;
}

/* Visualizers */
.visualizers {
  display: flex;
  justify-content: space-around;
  margin-bottom: 30px;
  gap: 20px;
  flex-wrap: wrap;
}

.viz-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.viz-label {
  font-size: 0.9rem;
  color: #636e72;
  margin-bottom: 8px;
  font-weight: 600;
}

canvas {
  background: #f8f9fa;
  border: 1px solid #dfe6e9;
  border-radius: 8px;
  box-shadow: inset 0 2px 5px rgba(0,0,0,0.05);
}

/* Controls */
.controls {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.control-item {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.control-item label {
  font-weight: bold;
  color: #636e72;
}

input[type="range"] {
  width: 100%;
  accent-color: #6c5ce7;
}

.model-select {
  width: 100%;
  padding: 12px;
  border: 2px solid #dfe6e9;
  border-radius: 12px;
  font-size: 1rem;
  background-color: white;
  cursor: pointer;
  transition: all 0.3s ease;
}

.model-select:focus {
  border-color: #6c5ce7;
  outline: none;
}

.btn-realtime {
  padding: 15px;
  border: 2px solid #6c5ce7;
  border-radius: 12px;
  background: transparent;
  color: #6c5ce7;
  font-size: 1.1rem;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-realtime:hover {
  background: #f9f9ff;
}

.btn-realtime.active {
  background: #ff7675;
  border-color: #ff7675;
  color: white;
}

.status {
  margin-top: 15px;
  display: flex;
  align-items: center;
  gap: 10px;
  color: #00b894;
  font-weight: bold;
}

.pulse {
  display: inline-block;
  width: 10px;
  height: 10px;
  background: #00b894;
  border-radius: 50%;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.5); opacity: 0.5; }
  100% { transform: scale(1); opacity: 1; }
}

footer {
  text-align: center;
  margin-top: 60px;
  font-size: 0.9rem;
  color: #b2bec3;
}
</style>