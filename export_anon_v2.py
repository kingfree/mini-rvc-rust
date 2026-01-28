import os
import torch
import onnx
from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_ONNX import SynthesizerTrnMs768NSFsid_ONNX

def export_anon():
    model_path = "models/Anon_v2.pth"
    output_path = "models/Anon_v2.onnx"
    merged_path = "models/Anon_v2_merged.onnx"
    
    print(f"Loading {model_path}...")
    cpt = torch.load(model_path, map_location="cpu")
    
    net_g_onnx = SynthesizerTrnMs768NSFsid_ONNX(*cpt['config'], is_half=False)
    net_g_onnx.load_state_dict(cpt['weight'], strict=False)
    net_g_onnx.eval()
    
    # Dummy inputs
    feats = torch.randn(1, 64, 768)
    p_len = torch.LongTensor([64])
    pitch = torch.zeros(1, 64, dtype=torch.long)
    pitchf = torch.zeros(1, 64, dtype=torch.float)
    sid = torch.LongTensor([0])
    
    print(f"Exporting to {output_path} with opset 17...")
    # Trying opset 17 to see if Pad operator is handled differently/better for Candle
    torch.onnx.export(
        net_g_onnx,
        (feats, p_len, pitch, pitchf, sid),
        output_path,
        opset_version=17, 
        do_constant_folding=True,
        input_names=["feats", "p_len", "pitch", "pitchf", "sid"],
        output_names=["audio"]
    )
    
    print("Merging weights...")
    model = onnx.load(output_path)
    onnx.save_model(model, merged_path, save_as_external_data=False)
    print(f"Export success: {merged_path}")

if __name__ == "__main__":
    export_anon()
