import os
import sys
import torch
import onnx

# Add voice-changer server to path
sys.path.append("/home/mei/dev/voice-changer/server")

from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_ONNX import (
    SynthesizerTrnMs768NSFsid_ONNX,
)

def export(input_path, output_path):
    print(f"Loading {input_path}...")
    cpt = torch.load(input_path, map_location="cpu")
    
    # RVC v2 uses 768 channels
    # config is usually (spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsched_initial_channel, upsched_resblock_kernel_sizes, upsched_resblock_dilation_sizes)
    
    config = cpt["config"]
    print(f"Config: {config}")
    
    net_g_onnx = SynthesizerTrnMs768NSFsid_ONNX(*config, is_half=False)
    net_g_onnx.load_state_dict(cpt["weight"], strict=False)
    net_g_onnx.eval()
    
    # Inputs: feats (1, T, 768), p_len (1), pitch (1, T), pitchf (1, T), sid (1)
    # RVC v2 embed channels = 768
    emb_channels = 768
    test_len = 64
    
    feats = torch.randn(1, test_len, emb_channels)
    p_len = torch.LongTensor([test_len])
    pitch = torch.zeros(1, test_len, dtype=torch.long)
    pitchf = torch.zeros(1, test_len, dtype=torch.float)
    sid = torch.LongTensor([0])
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        net_g_onnx,
        (feats, p_len, pitch, pitchf, sid),
        output_path,
        opset_version=12,
        do_constant_folding=True,
        input_names=["feats", "p_len", "pitch", "pitchf", "sid"],
        output_names=["audio"],
        # dynamic_axes={
        #     "feats": {1: "num_frames"},
        #     "pitch": {1: "num_frames"},
        #     "pitchf": {1: "num_frames"},
        # },
    )
    
    # print("Simplifying...")
    # model_onnx = onnx.load(output_path)
    # model_simp, check = simplify(model_onnx)
    # assert check, "Simplified ONNX model could not be validated"
    # onnx.save(model_simp, output_path.replace(".onnx", "_simple.onnx"))
    print("Done!")

if __name__ == "__main__":
    export("/home/mei/dev/mini-rvc-rust/models/Yukina_v2.pth", "/home/mei/dev/mini-rvc-rust/models/Yukina_v2.onnx")
