use tract_onnx::prelude::*;

fn main() -> anyhow::Result<()> {
    let model_path = "models/Anon_v2_merged.onnx";
    if !std::path::Path::new(model_path).exists() {
        println!("Model not found: {}", model_path);
        return Ok(());
    }
    println!("Checking model: {}", model_path);
    let model = tract_onnx::onnx().model_for_path(model_path)?;
    
    println!("Inputs: {:?}", model.input_outlets());
    println!("Outputs: {:?}", model.output_outlets());
    
    for outlet in model.output_outlets()? {
        println!("Output node: {}", model.node(outlet.node).name);
    }
    
    for node in model.nodes() {
        println!("Node: {}, Op: {:?}", node.name, node.op);
    }
    
    Ok(())
}
