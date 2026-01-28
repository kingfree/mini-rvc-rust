import faiss
import numpy as np
from safetensors.numpy import save_file
import sys
import os

def convert_index(index_path, output_path):
    print(f"Loading index from {index_path}...")
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        print(f"Error reading index: {e}")
        return

    print(f"Index type: {type(index)}")
    print(f"ntotal: {index.ntotal}")
    print(f"d: {index.d}")

    if index.ntotal == 0:
        print("Index is empty.")
        return

    # Extract vectors
    # If it's an IVF index, we might need to reconstruct.
    # index.reconstruct_n(0, ntotal) usually works for indices that store full vectors (like IVFFlat).
    
    try:
        print("Extracting vectors...")
        # reconstruct_n returns float32 numpy array
        vectors = index.reconstruct_n(0, index.ntotal)
        print(f"Extracted shape: {vectors.shape}")
        
        # Save as safetensors
        tensors = {
            "vectors": vectors
        }
        save_file(tensors, output_path)
        print(f"Saved to {output_path}")
        
    except Exception as e:
        print(f"Error extracting vectors: {e}")
        # Fallback for some index types?
        if hasattr(index, "make_direct_map"):
            try:
                print("Attempting make_direct_map...")
                index.make_direct_map()
                vectors = index.reconstruct_n(0, index.ntotal)
                tensors = {"vectors": vectors}
                save_file(tensors, output_path)
                print(f"Saved to {output_path}")
            except Exception as e2:
                print(f"Fallback failed: {e2}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_index.py <index_path> <output_path>")
        # Default for testing
        idx_path = "models/Yukina_v2.index"
        out_path = "models/Yukina_v2_index.safetensors"
        if os.path.exists(idx_path):
            convert_index(idx_path, out_path)
    else:
        convert_index(sys.argv[1], sys.argv[2])
