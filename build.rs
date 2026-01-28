use burn_import::onnx::ModelGen;

fn main() {
    // 基础构建逻辑：将 ONNX 模型转换为纯 Rust 代码
    // 我们使用了自定义修复后的模型，确保节点拓扑顺序正确
    ModelGen::new()
        .input("/home/mei/dev/mini-rvc-rust/model_fixed_v3.onnx")
        .out_dir("model")
        .run_from_script();
}
