// 声明生成的模型模块
// 注意：在实际构建成功前，这个 include 可能会导致 IDE 报错
// mod model {
//     include!(concat!(env!("OUT_DIR"), "/model/model_fixed_v3.rs"));
// }

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("--- Mini RVC Rust (Burn 实现) ---");
    println!("目标：纯 Rust 环境，无 C FFI 依赖");

    // 1. 初始化后端 (使用 WGPU 或 NdArray)
    // 这里先以 NdArray (CPU) 为例，因为它最通用
    println!("正在初始化推理后端...");

    // 2. 加载模型逻辑 (待模型代码生成成功后启用)
    println!("提示：当前处于代码生成实验阶段。");
    println!("一旦 build.rs 成功将 ONNX 转换为 Rust 代码，");
    println!("我们就可以在这里实例化模型并进行实时语音转换。");

    Ok(())
}
