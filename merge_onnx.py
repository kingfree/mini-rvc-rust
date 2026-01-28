import onnx
from onnx.external_data_helper import load_external_data_for_model, convert_model_to_external_data

def merge(input_path, output_path):
    print(f"Merging {input_path}...")
    model = onnx.load(input_path)
    # load_external_data_for_model(model, ".") # This is implicit if file is there
    
    print("Saving to single file...")
    onnx.save_model(model, output_path, save_as_external_data=False)
    print("Done!")

if __name__ == "__main__":
    merge("/home/mei/dev/mini-rvc-rust/models/Yukina_v2.onnx", "/home/mei/dev/mini-rvc-rust/models/Yukina_v2_merged.onnx")
