<template>
  <div class="mashutan-app">
    <header>
      <h1>🔮 Mashutan RVC 占卜面板</h1>
      <p class="subtitle">让你的声音在 Rust 的魔力中焕发新生 ✨</p>
    </header>

    <main>
      <section class="card upload-section">
        <h2>🎙️ 声音采样</h2>
        <div 
          class="drop-zone" 
          @dragover.prevent 
          @drop.prevent="handleDrop"
          @click="triggerFileInput"
        >
          <input 
            type="file" 
            ref="fileInputRef" 
            style="display: none" 
            accept=".wav"
            @change="handleFileSelect"
          >
          <div v-if="!selectedFile">
            <span class="icon">📂</span>
            <p>拖拽 .wav 文件到这里，或者点击上传</p>
          </div>
          <div v-else>
            <span class="icon">📄</span>
            <p>{{ selectedFile.name }}</p>
          </div>
        </div>
        <button 
          class="btn-primary" 
          :disabled="!selectedFile || processing"
          @click="startExtraction"
        >
          {{ processing ? '🔮 正在感应特征...' : '一语道破！(开始提取)' }}
        </button>
      </section>

      <section v-if="result" class="card result-section">
        <h2>✨ 占卜结果 (特征提取)</h2>
        <div class="result-details">
          <div class="item">
            <span class="label">状态:</span>
            <span class="value success">✅ {{ result.message }}</span>
          </div>
          <div class="item">
            <span class="label">推理耗时:</span>
            <span class="value">{{ result.time_ms }} ms</span>
          </div>
          <div class="item">
            <span class="label">输出维度:</span>
            <span class="value">{{ result.shape.join(' x ') }}</span>
          </div>
        </div>
      </section>
    </main>

    <footer>
      <p>由 玛修坦 & 森亚露露卡 强力驱动 | Powered by Candle & Rust 🦀</p>
    </footer>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const selectedFile = ref<File | null>(null)
const fileInputRef = ref<HTMLInputElement | null>(null)
const processing = ref(false)
const result = ref<{ success: boolean; message: string; time_ms: number; shape: number[] } | null>(null)

const triggerFileInput = () => {
  fileInputRef.value?.click()
}

const handleFileSelect = (e: Event) => {
  const target = e.target as HTMLInputElement
  if (target.files && target.files.length > 0) {
    selectedFile.value = target.files[0]
  }
}

const handleDrop = (e: DragEvent) => {
  if (e.dataTransfer && e.dataTransfer.files.length > 0) {
    selectedFile.value = e.dataTransfer.files[0]
  }
}

const startExtraction = async () => {
  if (!selectedFile.value) return
  
  processing.value = true
  result.value = null
  
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  
  try {
    const response = await fetch('/api/extract', {
      method: 'POST',
      body: formData
    })
    result.value = await response.json()
  } catch (e) {
    console.error(e)
    alert('🔮 哎呀，占卜水晶球碎了（请求失败）')
  } finally {
    processing.value = false
  }
}
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

.drop-zone {
  border: 2px dashed #dfe6e9;
  border-radius: 15px;
  padding: 40px;
  text-align: center;
  cursor: pointer;
  margin-bottom: 20px;
  transition: all 0.3s ease;
}

.drop-zone:hover {
  border-color: #6c5ce7;
  background: #f9f9ff;
}

.icon {
  font-size: 3rem;
  display: block;
  margin-bottom: 10px;
}

.btn-primary {
  width: 100%;
  padding: 15px;
  border: none;
  border-radius: 12px;
  background: #6c5ce7;
  color: white;
  font-size: 1.1rem;
  font-weight: bold;
  cursor: pointer;
  transition: background 0.3s ease;
}

.btn-primary:hover:not(:disabled) {
  background: #5b4cc4;
}

.btn-primary:disabled {
  background: #b2bec3;
  cursor: not-allowed;
}

.result-details .item {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid #f1f2f6;
}

.label {
  font-weight: bold;
  color: #636e72;
}

.success {
  color: #00b894;
}

footer {
  text-align: center;
  margin-top: 60px;
  font-size: 0.9rem;
  color: #b2bec3;
}
</style>
