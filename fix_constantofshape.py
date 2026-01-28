#!/usr/bin/env python3
"""
Fix ConstantOfShape nodes to explicitly use FP32
"""
import onnx
from onnx import helper, numpy_helper
import numpy as np

model = onnx.load("models/Yukina_v2_tract.onnx")
graph = model.graph

print("Fixing ConstantOfShape nodes...")

for node in graph.node:
    if node.op_type == 'ConstantOfShape':
        # Check if it has a value attribute
        has_value = False
        for attr in node.attribute:
            if attr.name == 'value':
                has_value = True
                if attr.t.data_type != onnx.TensorProto.FLOAT:
                    print(f"  {node.name}: setting dtype to FLOAT")
                    attr.t.data_type = onnx.TensorProto.FLOAT
                break

        # If no value attribute, add one with FP32 zero
        if not has_value:
            print(f"  {node.name}: adding FP32 zero value")
            zero_tensor = numpy_helper.from_array(np.array([0.0], dtype=np.float32), '')
            node.attribute.append(
                helper.make_attribute('value', zero_tensor)
            )

# Save
output_path = "models/Yukina_v2_tract.onnx"
onnx.save(model, output_path)
print(f"\nSaved to: {output_path}")

try:
    onnx.checker.check_model(model)
    print("✓ Valid")
except Exception as e:
    print(f"Check: {e}")
