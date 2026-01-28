import onnx

model_path = "pretrain/content_vec_500.onnx"
# model_path = "pretrain/content_vec_500.onnx"
try:
    model = onnx.load(model_path)
    print("Inputs:")
    for input in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else "Dyn" for d in input.type.tensor_type.shape.dim]
        print(f"  {input.name}: {shape}")

    print("Initializers:")
    for init in model.graph.initializer:
        print(f"  {init.name}: {init.data_type}")
except Exception as e:
    print(e)
