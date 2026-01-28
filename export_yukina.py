import os
import sys
import torch
import onnx

# Add voice-changer to path
sys.path.append(os.path.abspath("../voice-changer/server"))

from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_ONNX import SynthesizerTrnMs768NSFsid_ONNX

def export_yukina():
    model_path = "models/Yukina_v2.pth"
    output_path = "models/Yukina_v2.onnx"
    merged_path = "models/Yukina_v2_merged.onnx"
    
    print(f"Loading {model_path}...")
    cpt = torch.load(model_path, map_location="cpu")
    
    # Yukina v2 is usually RVC v2, 40k or 48k. 
    # Config is inside the pth usually, or we need to guess parameters.
    # SynthesizerTrnMs768NSFsid_ONNX expects *cpt['config'] if it exists.
    # Let's check keys first if this fails.
    
    if 'config' not in cpt:
        # Fallback config if not in pth (RVC v2 40k default)
        # spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, n_speakers, gin_channels, use_spectral_norm, use_sdp
        # These are standard RVC v2 40k params usually.
        print("Warning: Config not found in pth, using defaults or failing.")
        # Try to use what's in 'params' if available?
    
    net_g_onnx = SynthesizerTrnMs768NSFsid_ONNX(*cpt['config'], is_half=False)
    net_g_onnx.load_state_dict(cpt['weight'], strict=False)
    net_g_onnx.eval()
    
    # Dummy inputs for RVC v2
    # feats: [1, 128, 768] (Matching our streaming chunk size)
    feats = torch.randn(1, 128, 768)
    p_len = torch.LongTensor([128])
    pitch = torch.zeros(1, 128, dtype=torch.long)
    pitchf = torch.zeros(1, 128, dtype=torch.float)
    sid = torch.LongTensor([0])
    
    print(f"Exporting to {output_path} with opset 17...")
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
    export_yukina()