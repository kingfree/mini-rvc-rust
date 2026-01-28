# 实时 RVC 语音转换 - 进度记录

## 当前状态

### 已完成
- ContentVec 特征提取 (Tract) - 工作正常 (~400ms in Release)
- RMVPE 音高提取 (Tract) - 工作正常 (~100ms in Release)
- RVC 模型推理 (Candle) - **验证通过 & Metal 加速**
  - 解决了 `ConvTranspose1d` 和设备不匹配的所有问题。
  - 性能：15s 音频仅需 ~7.8s 推理 (RTF ~0.5)。
- **实时管道集成**:
  - 特征线性插值 (50fps -> 100fps)。
  - 输出重采样 (40kHz -> 16kHz) 使用 `rubato`。
  - SOLA 拼接与流式裁剪逻辑。
- 基础 HTTP & WebSocket 实时服务器。
- 前端实时流 UI (波形展示 + 模型选择)。
- **端到端测试**:
  - `test_full` 二进制验证了完整管道 (音频加载 -> 特征 -> 推理 -> 重采样 -> 保存)。
  - 修复了 `tract` 动态维度推理问题 (通过使用 concrete input fact `[1, 160, 128]`)。

### 待完成阶段
1. **阶段 1** - 整体测试
   - 启动服务器和前端，测试闭环实时效果。
2. **阶段 2** - 延迟优化
   - 目前采用 1s 窗口 + 0.5s 步进，存在 0.5s 固有延迟。
   - 尝试减小窗口大小 (e.g. 0.5s 窗口 + 0.2s 步进) 以降低延迟。
3. **阶段 3** - 模型多样性
   - 导出更多角色模型进行测试。

## 关键文件
- `src/realtime_pipeline.rs`: 核心推理逻辑与 DSP 处理。
- `src/main.rs`: WebSocket 服务与状态管理。
- `web/src/App.vue`: 前端 UI 与 AudioWorklet 调度。