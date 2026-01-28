<template>
  <div class="vc-app">
    <header class="vc-header">
      <div class="logo-area">
        <span class="logo-icon">🦀</span>
        <h1>Mini RVC Rust (实时变声)</h1>
      </div>
      <div class="status-bar">
        <div :class="['status-indicator', isRealtime ? 'online' : 'offline']">
          {{ isRealtime ? '运行中' : '已停止' }}
        </div>
        <div class="latency-pill" v-if="isRealtime">
          延迟: <span>{{ latency }}ms</span>
        </div>
      </div>
    </header>

    <main class="vc-main">
      <div class="control-grid">
        <!-- Character / Model Section -->
        <section class="vc-card model-card">
          <div class="card-header">
            <h3>👤 角色 / 模型</h3>
          </div>
          <div class="model-selector">
            <select v-model="selectedModel" @change="changeConfig" class="vc-select">
              <option v-for="model in models" :key="model.id" :value="model.id">
                {{ model.name }}
              </option>
            </select>
          </div>
          <div class="avatar-area">
            <div class="avatar-placeholder">
              {{ selectedModel ? selectedModel.charAt(0).toUpperCase() : '?' }}
            </div>
          </div>
        </section>

        <!-- Audio Settings Section -->
        <section class="vc-card settings-card">
          <div class="card-header">
            <h3>🎚️ 音频控制</h3>
          </div>
          
          <div class="control-group">
            <div class="slider-item">
              <div class="slider-label">
                <span>音高偏移 (Pitch)</span>
                <span class="value">{{ pitchShift }}</span>
              </div>
              <input type="range" min="-12" max="12" step="1" v-model="pitchShift" @input="changeConfig">
            </div>

            <div class="slider-item">
              <div class="slider-label">
                <span>检索比例 (Index)</span>
                <span class="value">{{ indexRate.toFixed(2) }}</span>
              </div>
              <input type="range" min="0" max="1" step="0.01" v-model="indexRate" @input="changeConfig">
            </div>

            <div class="slider-item">
              <div class="slider-label">
                <span>输入增益 (Gain)</span>
                <span class="value">{{ inputGain.toFixed(1) }}x</span>
              </div>
              <input type="range" min="0" max="5" step="0.1" v-model="inputGain" @input="updateWorkletParams">
            </div>

            <div class="slider-item">
              <div class="slider-label">
                <span>输出增益 (Gain)</span>
                <span class="value">{{ outputGain.toFixed(1) }}x</span>
              </div>
              <input type="range" min="0" max="5" step="0.1" v-model="outputGain" @input="updateWorkletParams">
            </div>

            <div class="slider-item">
              <div class="slider-label">
                <span>静音阈值 (Gate)</span>
                <span class="value">{{ threshold }} dB</span>
              </div>
              <input type="range" min="-100" max="0" step="1" v-model="threshold" @input="updateWorkletParams">
            </div>
          </div>
        </section>

        <!-- Visualization / Output -->
        <section class="vc-card viz-card">
          <div class="viz-wrapper">
            <div class="viz-box">
              <div class="viz-header">输入波形 (Input)</div>
              <canvas ref="inputCanvasRef" width="400" height="80"></canvas>
            </div>
            <div class="viz-box">
              <div class="viz-header">输出波形 (Output)</div>
              <canvas ref="outputCanvasRef" width="400" height="80"></canvas>
            </div>
          </div>
          
          <button 
            :class="['vc-btn-main', isRealtime ? 'stop' : 'start']"
            @click="toggleRealtime"
          >
            {{ isRealtime ? '停止实时变换' : '开启实时变换' }}
          </button>
        </section>
      </div>
    </main>

    <footer class="vc-footer">
      由 Candle & Tract 强力驱动 | 高性能实时语音转换系统 ⚡
    </footer>

  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const isRealtime = ref(false)
const pitchShift = ref(0)
const indexRate = ref(0.3)
const inputGain = ref(1.0)
const outputGain = ref(1.0)
const threshold = ref(-60)
const models = ref<{id: string, name: string}[]>([])
const selectedModel = ref<string>('')
const latency = ref(0)

let audioContext: AudioContext | null = null
let ws: WebSocket | null = null
let processorNode: AudioWorkletNode | null = null
let source: MediaStreamAudioSourceNode | null = null

// Visualization
const inputCanvasRef = ref<HTMLCanvasElement | null>(null)
const outputCanvasRef = ref<HTMLCanvasElement | null>(null)
let inputAnalyser: AnalyserNode | null = null
let outputAnalyser: AnalyserNode | null = null
let animationId: number | null = null

onMounted(async () => {
  try {
    const res = await fetch('/api/models')
    models.value = await res.json()
    if (models.value.length > 0) selectedModel.value = models.value[0].id
  } catch (e) {
    console.error("Failed to load models", e)
  }
})

const changeConfig = () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ 
      pitch_shift: Number(pitchShift.value),
      index_rate: Number(indexRate.value),
      model_id: selectedModel.value 
    }))
  }
}

const updateWorkletParams = () => {
  if (processorNode) {
    const inputGainParam = processorNode.parameters.get('inputGain')
    const outputGainParam = processorNode.parameters.get('outputGain')
    const thresholdParam = processorNode.parameters.get('threshold')
    
    if (inputGainParam) inputGainParam.value = Number(inputGain.value)
    if (outputGainParam) outputGainParam.value = Number(outputGain.value)
    if (thresholdParam) thresholdParam.value = Number(threshold.value)
  }
}

const draw = () => {
  if (!isRealtime.value) return
  
  const render = (analyser: AnalyserNode | null, canvas: HTMLCanvasElement | null, color: string) => {
    if (!analyser || !canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    analyser.getByteTimeDomainData(dataArray)
    
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.lineWidth = 2
    ctx.strokeStyle = color
    ctx.beginPath()
    const sliceWidth = canvas.width / bufferLength
    let x = 0
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0
      const y = v * canvas.height / 2
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
      x += sliceWidth
    }
    ctx.stroke()
  }

  render(inputAnalyser, inputCanvasRef.value, '#00d1b2')
  render(outputAnalyser, outputCanvasRef.value, '#ffdd57')
  animationId = requestAnimationFrame(draw)
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
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`)
    ws.binaryType = 'arraybuffer'

    processorNode = new AudioWorkletNode(audioContext, 'audio-processor')
    updateWorkletParams() // Initial params

    source = audioContext.createMediaStreamSource(stream)
    inputAnalyser = audioContext.createAnalyser()
    outputAnalyser = audioContext.createAnalyser()
    inputAnalyser.fftSize = 512
    outputAnalyser.fftSize = 512

    source.connect(inputAnalyser)
    inputAnalyser.connect(processorNode)
    processorNode.connect(outputAnalyser)
    outputAnalyser.connect(audioContext.destination)

    processorNode.port.onmessage = (e) => {
      if (e.data.type === 'audio-input' && ws?.readyState === WebSocket.OPEN) {
        ws.send(e.data.samples.buffer)
      }
    }

    ws.onmessage = (e) => {
      if (e.data instanceof ArrayBuffer) {
        processorNode?.port.postMessage({ type: 'audio-output', samples: new Float32Array(e.data) })
      } else {
        const meta = JSON.parse(e.data)
        if (meta.latency) latency.value = meta.latency
      }
    }

    ws.onopen = () => {
      isRealtime.value = true
      changeConfig()
      draw()
    }
    ws.onclose = stopRealtime
  } catch (e) {
    alert('Failed to start audio: ' + e)
    stopRealtime()
  }
}

const stopRealtime = () => {
  isRealtime.value = false
  if (animationId) cancelAnimationFrame(animationId)
  ws?.close()
  audioContext?.close()
  source = null; processorNode = null; ws = null; audioContext = null;
}

onUnmounted(stopRealtime)
</script>

<style scoped>
.vc-app {
  min-height: 100vh;
  background-color: #f0f2f5;
  color: #2d3436;
  font-family: 'Inter', -apple-system, sans-serif;
  padding: 20px;
}

.vc-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  border-bottom: 1px solid #dfe6e9;
  padding-bottom: 15px;
  background: white;
  padding: 15px 25px;
  border-radius: 15px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}

.logo-area {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon { font-size: 2rem; }
h1 { font-size: 1.5rem; font-weight: 700; color: #2d3436; margin: 0; }

.status-bar {
  display: flex;
  align-items: center;
  gap: 15px;
}

.status-indicator {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 800;
  background: #f1f2f6;
  color: #636e72;
}

.status-indicator.online { color: #00b894; border: 1px solid #00b894; background: #e6fffb; }
.status-indicator.offline { color: #ff7675; border: 1px solid #ff7675; background: #fff5f5; }

.latency-pill {
  font-size: 0.85rem;
  color: #636e72;
}
.latency-pill span { color: #6c5ce7; font-weight: bold; }

.control-grid {
  display: grid;
  grid-template-columns: 280px 1fr;
  grid-template-rows: auto auto;
  gap: 20px;
  max-width: 1000px;
  margin: 0 auto;
}

.vc-card {
  background: white;
  border-radius: 15px;
  padding: 25px;
  border: 1px solid #e1e4e8;
  box-shadow: 0 10px 25px rgba(0,0,0,0.03);
}

.card-header h3 {
  margin: 0 0 20px 0;
  font-size: 1rem;
  color: #636e72;
  text-transform: uppercase;
  letter-spacing: 1px;
  border-left: 4px solid #6c5ce7;
  padding-left: 10px;
}

/* Model Card */
.model-selector { margin-bottom: 20px; }
.vc-select {
  width: 100%;
  background: #fff;
  border: 2px solid #dfe6e9;
  color: #2d3436;
  padding: 12px;
  border-radius: 10px;
  outline: none;
  font-size: 1rem;
  transition: border-color 0.2s;
}
.vc-select:focus { border-color: #6c5ce7; }

.avatar-area {
  display: flex;
  justify-content: center;
  padding: 10px 0;
}

.avatar-placeholder {
  width: 100px;
  height: 100px;
  background: linear-gradient(135deg, #6c5ce7, #a29bfe);
  border-radius: 20%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.5rem;
  font-weight: bold;
  color: white;
  box-shadow: 0 10px 20px rgba(108, 92, 231, 0.2);
}

/* Settings Card */
.settings-card {
  grid-row: span 2;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.slider-item {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.slider-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.9rem;
  font-weight: 600;
  color: #2d3436;
}

.slider-label .value { color: #6c5ce7; font-weight: 800; }

input[type="range"] {
  width: 100%;
  accent-color: #6c5ce7;
  height: 6px;
  background: #dfe6e9;
  border-radius: 3px;
  appearance: none;
}

/* Visualizer Card */
.viz-card {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.viz-wrapper {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.viz-box {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 15px;
  border: 1px solid #edf2f7;
}

.viz-header {
  font-size: 0.75rem;
  font-weight: 700;
  color: #b2bec3;
  margin-bottom: 8px;
  text-align: center;
}

canvas { width: 100%; height: 80px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.05)); }

.vc-btn-main {
  width: 100%;
  padding: 18px;
  border-radius: 12px;
  border: none;
  font-weight: 800;
  font-size: 1.1rem;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.vc-btn-main.start { background: #6c5ce7; color: #fff; }
.vc-btn-main.stop { background: #ff7675; color: #fff; }
.vc-btn-main:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(108, 92, 231, 0.3); }
.vc-btn-main:active { transform: translateY(0); }

.vc-footer {
  text-align: center;
  margin-top: 40px;
  color: #b2bec3;
  font-size: 0.85rem;
}

@media (max-width: 800px) {
  .control-grid { grid-template-columns: 1fr; }
  .settings-card { grid-row: auto; }
}
</style>
