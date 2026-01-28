use tract_onnx::prelude::*;

fn main() -> anyhow::Result<()> {
    let model_path = "pretrain/content_vec_500.onnx";
    if !std::path::Path::new(model_path).exists() {
        println!("Model not found: {}", model_path);
        return Ok(());
    }
    println!("Checking model: {}", model_path);
    let model = tract_onnx::onnx().model_for_path(model_path)?;
    
    for node in model.nodes() {
        println!("Node: {}, Op: {:?}", node.name, node.op);
    }
    
    Ok(())
}
