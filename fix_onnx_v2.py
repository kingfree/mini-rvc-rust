import onnx

m_path = '/home/mei/dev/voice-changer/server/model_dir/0/tsukuyomi_v2_40k_e100_simple.onnx'
o_path = '/home/mei/dev/mini-rvc-rust/model_fixed.onnx'

m = onnx.load(m_path)
import onnx.shape_inference
m_inf = onnx.shape_inference.infer_shapes(m)
onnx.save(m_inf, o_path)
print(f"Model topological check done: {o_path}")
