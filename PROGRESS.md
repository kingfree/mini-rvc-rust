# 实时 RVC 语音转换 - 进度记录

## 当前状态

### 已完成
- ContentVec 特征提取 (Tract) - 工作正常
- RMVPE 音高提取 (Tract) - 工作正常
- RVC 模型推理 (Candle) - **验证通过 & Metal 加速**
- **实时管道集成**:
  - 特征线性插值 (50fps -> 100fps)。
  - 输出重采样 (40kHz -> 16kHz) 使用 `rubato`。
  - SOLA 拼接与流式裁剪逻辑。
- **高性能架构**:
  - **多线程拆分**: 推理逻辑在独立线程中运行，不阻塞 WebSocket 收发。
  - **延迟测量**: 实现了端到端的延迟监控。
- **Index (检索) 支持**:
  - 实现了基于 SafeTensors 的特征检索逻辑 (Metal 加速)。
  - 支持动态调整 `index_rate`。
- 基础 HTTP & WebSocket 实时服务器。
- 前端实时流 UI (波形展示 + 模型选择)。

### 性能数据 (Release Mode + Metal)
- RVC Inference: ~8s (for 15s audio, RTF ~0.5)。
- 固有延迟: ~300ms (基于 400ms 窗口 + 300ms 步进)。

### 待完成阶段
1. **阶段 1** - 模型多样性
   - 导出更多角色模型并配备对应的 `.index` (SafeTensors) 文件。
2. **阶段 2** - 鲁棒性优化
   - 处理网络抖动对音频流的影响。

## 关键文件
- `src/realtime_pipeline.rs`: 核心推理逻辑、检索、重采样。
- `src/main.rs`: 多线程调度、WebSocket 服务。
- `convert_index.py`: FAISS 索引转换工具。
