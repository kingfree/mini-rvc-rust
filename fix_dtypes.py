#!/usr/bin/env python3
"""
Fix ONNX model to use FP32 everywhere
"""
import onnx
from onnx import numpy_helper
import numpy as np

model = onnx.load("models/Yukina_v2_norandom.onnx")
graph = model.graph

print("Fixing data types to FP32...")

# Fix initializers
for init in graph.initializer:
    if init.data_type == onnx.TensorProto.DOUBLE:
        print(f"  Initializer {init.name}: DOUBLE -> FLOAT")
        array = numpy_helper.to_array(init)
        new_init = numpy_helper.from_array(array.astype(np.float32), init.name)
        graph.initializer.remove(init)
        graph.initializer.append(new_init)

# Fix node attributes (especially ConstantOfShape)
for node in graph.node:
    if node.op_type == 'ConstantOfShape':
        for attr in node.attribute:
            if attr.name == 'value' and attr.t.data_type == onnx.TensorProto.DOUBLE:
                print(f"  Node {node.name}: value DOUBLE -> FLOAT")
                array = numpy_helper.to_array(attr.t)
                new_tensor = numpy_helper.from_array(array.astype(np.float32), '')
                attr.t.CopyFrom(new_tensor)

    if node.op_type == 'Constant':
        for attr in node.attribute:
            if attr.name == 'value' and attr.t.data_type == onnx.TensorProto.DOUBLE:
                print(f"  Node {node.name}: value DOUBLE -> FLOAT")
                array = numpy_helper.to_array(attr.t)
                new_tensor = numpy_helper.from_array(array.astype(np.float32), '')
                attr.t.CopyFrom(new_tensor)

# Fix Cast nodes
for node in graph.node:
    if node.op_type == 'Cast':
        for attr in node.attribute:
            if attr.name == 'to' and attr.i == onnx.TensorProto.DOUBLE:
                print(f"  Cast node {node.name}: target DOUBLE -> FLOAT")
                attr.i = onnx.TensorProto.FLOAT

# Save
output_path = "models/Yukina_v2_tract.onnx"
onnx.save(model, output_path)
print(f"\nSaved to: {output_path}")

# Check
try:
    onnx.checker.check_model(model)
    print("✓ Model is valid")
except Exception as e:
    print(f"Check: {e}")
