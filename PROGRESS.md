# 实时 RVC 语音转换 - 进度记录

## 当前状态

### 已完成
- ContentVec 特征提取 (Tract) - 工作正常, ~830ms
- RMVPE 音高提取 (Tract) - 工作正常, ~290ms
- 音频重采样、Mel 频谱提取
- 基础 HTTP 服务器 (Axum)
- 基础 Vue 前端脚手架
- 模型内存释放优化（用完即释放 ContentVec/RMVPE）

### 进行中 - RVC 模型推理 (Candle)
**问题：** 在 3.8GB 内存的机器上 OOM (exit code 144)

**已做的修复（在 ~/dev/candle）：**
1. `candle-onnx/src/eval.rs` - ScatterND 操作 `update_slice` 在非标量情况下未 squeeze，导致 `slice_scatter` rank 不匹配
2. `candle-onnx/src/eval.rs` - 给 `simple_eval_` 添加引用计数，中间张量用完即释放

**输入形状确认（通过 ONNX 检查）：**
- feats: [1, 64, 768] (f32)
- p_len: [1] (i64)
- pitch: [1, 64] (i64)
- pitchf: [1, 64] (f32)
- sid: [1] (i64)

**仍然 OOM 的原因：**
- `candle_onnx::read_file` 加载整个 protobuf (~109MB) 到内存
- `simple_eval` 中 initializer 权重全部加载
- 即使加了引用计数，峰值内存仍然太高
- 需要在有更多内存的机器上测试

### 待完成阶段
1. **阶段 1** - 验证 RVC 推理能跑通（需要更多内存的机器）
2. **阶段 2** - SOLA 音频拼接 (`src/audio_stitching.rs`)
3. **阶段 3** - WebSocket 实时流 (后端 + 前端 AudioWorklet)
4. **阶段 4** - 实时推理管道 (`src/realtime_pipeline.rs` + `src/ring_buffer.rs`)
5. **阶段 5** - 前端音频播放
6. **阶段 6** - 性能优化与测试
7. **阶段 7** - 配置和部署

## 关键文件

### 需要创建
- `src/audio_stitching.rs` - SOLA 算法
- `src/realtime_pipeline.rs` - 实时推理管道
- `src/ring_buffer.rs` - 环形缓冲区
- `web/public/audio-processor.js` - AudioWorklet 处理器
- `.env` - 环境配置

### 需要修改
- `src/main.rs` - WebSocket 路由和处理
- `web/src/App.vue` - 实时音频功能
- `web/vite.config.ts` - 代理配置
- `Cargo.toml` - 添加 ws feature

### 模型文件
- `models/Anon_v2_merged.onnx` (109MB)
- `pretrain/content_vec_500.onnx` (362MB)
- `pretrain/rmvpe.onnx` (345MB)

## Candle 修改记录 (~/dev/candle)

分支: `feat-onnx-ops-for-rvc`

### eval.rs 修改点
1. **ScatterND fix** (~行2783-2787): `flat_updates.narrow(0, i, 1)?` 后需要 `.squeeze(0)?`，否则 update_slice 多一个维度导致 slice_scatter rank 不匹配
2. **引用计数** (~行324-337, 2837-2849): 遍历图前统计每个值被引用次数，节点处理完后递减，为 0 且不是最终输出时从 values 中删除
