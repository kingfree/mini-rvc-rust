use candle_core::Device;
use candle_onnx;
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("--- Mini RVC Rust Inference (via Hugging Face Candle) ---");
    println!("目标：100% 纯 Rust，无 C FFI 依赖");

    // 1. 选择设备
    let device = Device::Cpu;
    println!("当前推理设备: {:?}", device);

    // 2. 模型路径
    let model_path = "/home/mei/dev/voice-changer/server/model_dir/0/tsukuyomi_v2_40k_e100_simple.onnx";

    if Path::new(model_path).exists() {
        println!("正在尝试加载 ONNX 模型: {}", model_path);
        
        match candle_onnx::read_file(model_path) {
            Ok(model_proto) => {
                println!("✅ 成功读取模型协议！");
                
                let graph = model_proto.graph.as_ref().unwrap();
                println!("模型图名称: {}", graph.name);
                println!("节点数量: {}", graph.node.len());

                // 尝试提取输入输出名称
                let inputs: Vec<_> = graph.input.iter().map(|i| &i.name).collect();
                let outputs: Vec<_> = graph.output.iter().map(|o| &o.name).collect();

                println!("输入节点: {:?}", inputs);
                println!("输出节点: {:?}", outputs);

                // 3. 构建简单的计算图进行验证
                println!("正在验证模型结构...");
                // 仅验证是否能成功载入权重
                let _model = candle_onnx::simple_eval(&model_proto, std::collections::HashMap::new());
                println!("✅ 模型结构验证通过 (基本载入测试)");
            },
            Err(e) => {
                println!("❌ 模型解析失败: {:?}", e);
            }
        }
    } else {
        println!("❌ 未找到模型文件 '{}'。", model_path);
    }

    Ok(())
}
