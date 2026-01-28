import onnx

model_path = "models/Yukina_v2_merged.onnx"
try:
    model = onnx.load(model_path)
    print("Inputs:")
    for input in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else "Dyn" for d in input.type.tensor_type.shape.dim]
        print(f"  {input.name}: {shape}")
except Exception as e:
    print(e)
