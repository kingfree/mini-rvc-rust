import onnx

def fix_topological_sort(model_path, output_path):
    # 重新加载并保存以尝试修复
    model = onnx.load(model_path)
    onnx.save(model, output_path)
    print(f"Model resaved to {output_path}")

fix_topological_sort('/home/mei/dev/voice-changer/server/model_dir/0/tsukuyomi_v2_40k_e100_simple.onnx', '/home/mei/dev/mini-rvc-rust/model_sorted.onnx')
